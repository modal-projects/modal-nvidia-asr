from __future__ import annotations

import os
import re
from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable, Any
import time
import math
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor


from nemo.collections.asr.inference.model_wrappers.asr_inference_wrapper import ASRInferenceWrapper
from nemo.collections.asr.inference.pipelines.pipeline_interface import PipelineInterface
from nemo.collections.asr.inference.streaming.buffering.audio_bufferer import BatchedAudioBufferer
from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.streaming.buffering.feature_bufferer import BatchedFeatureBufferer
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedRequestStreamer
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame, Request
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.state.state import StreamingState
from nemo.collections.asr.inference.streaming.text.text_processing import StreamingTextProcessor
from nemo.collections.asr.inference.utils.bpe_decoder import BPEDecoder
from nemo.collections.asr.inference.utils.context_manager import CacheAwareContextManager
from nemo.collections.asr.inference.utils.enums import RequestType
from nemo.collections.asr.inference.utils.pipeline_utils import (
    check_existance_of_required_attributes,
    get_leading_punctuation_regex_pattern,
    ids_to_text_without_stripping,
    get_confidence_utils,
)
from nemo.collections.asr.inference.utils.progressbar import ProgressBar
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.asr.inference.pipelines.base_pipeline import TranscribeStepOutput
from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import CacheAwareRNNTInferenceWrapper
from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_rnnt_decoder import RNNTGreedyDecoder
from nemo.collections.asr.inference.streaming.endpointing.greedy.greedy_rnnt_endpointing import RNNTGreedyEndpointing
from nemo.collections.asr.inference.streaming.state.cache_aware_rnnt_state import CacheAwareRNNTStreamingState
from nemo.collections.asr.inference.utils.endpointing_utils import millisecond_to_frames
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.inference.factory.buffered_pipeline_builder import BufferedPipelineBuilder
from nemo.collections.asr.inference.utils.enums import PipelineType
from nemo.collections.asr.inference.factory.base_builder import BaseBuilder
from nemo.collections.asr.inference.pipelines.cache_aware_ctc_pipeline import CacheAwareCTCPipeline
from nemo.collections.asr.inference.utils.enums import ASRDecodingType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer
    from nemo.collections.asr.inference.nmt.llm_translator import LLMTranslator

class BaseStreamingPipeline(PipelineInterface):
    """
    Base class for all pipelines.
    """

    def __init__(self):
        """Initialize state pool to store the state for each stream"""
        self._state_pool: dict[int, StreamingState] = {}

    def get_state(self, stream_id: int) -> StreamingState:
        """Retrieve state for a given stream ID."""
        return self._state_pool.get(stream_id, None)

    def get_states(self, stream_ids: Iterable[int]) -> list[StreamingState]:
        """Retrieve states for a list of stream IDs."""
        return [self.get_state(stream_id) for stream_id in stream_ids]

    def delete_state(self, stream_id: int) -> None:
        """Delete the state from the state pool."""
        if stream_id in self._state_pool:
            del self._state_pool[stream_id]

    def delete_states(self, stream_ids: Iterable[int]) -> None:
        """Delete states for a list of stream IDs."""
        for stream_id in stream_ids:
            self.delete_state(stream_id)

    def init_state(self, stream_id: int, options: ASRRequestOptions) -> StreamingState:
        """Initialize the state of the stream"""
        if stream_id not in self._state_pool:
            state = self.create_state(options)
            self._state_pool[stream_id] = state
        return self._state_pool[stream_id]

    def reset_session(self) -> None:
        """Reset the frame buffer and internal state pool"""
        self._state_pool.clear()

    def open_session(self) -> None:
        """Start a new session by resetting the internal state pool"""
        self.reset_session()

    def close_session(self) -> None:
        """Close the session by resetting the internal state pool"""
        self.reset_session()

    def open_streaming_session(self, options: ASRRequestOptions | None = None) -> int:
        """
        Open a new streaming session for real-time audio
        Args:
            options (ASRRequestOptions | None): Optional options for the stream
        Returns:
            int: The stream ID for this session
        """
        if not hasattr(self, '_streaming_request_generator'):
            raise RuntimeError("Pipeline not configured for streaming. Call init_streaming_request_generator() first.")
        
        # Create default options if none provided
        if options is None:
            options = ASRRequestOptions()
        
        return self._streaming_request_generator.open_stream(options)

    def append_streaming_audio(self, stream_id: int, audio_samples) -> None:
        """
        Append audio samples to an active streaming session
        Args:
            stream_id (int): The stream ID
            audio_samples: Audio samples (torch.Tensor or np.ndarray)
        """
        if not hasattr(self, '_streaming_request_generator'):
            raise RuntimeError("Pipeline not configured for streaming.")
        self._streaming_request_generator.append_audio(stream_id, audio_samples)

    def close_streaming_session(self, stream_id: int) -> None:
        """
        Mark a streaming session as ended (stops accepting new audio).
        State cleanup is deferred until all buffered frames are processed.
        
        NOTE: This method no longer immediately deletes state. Instead, it just
        marks the stream as ended in the request generator. The actual cleanup
        of _state_pool and context_manager happens later when the inference loop
        confirms all frames have been processed.
        
        Args:
            stream_id (int): The stream ID to close
        """
        # Close the stream in the request generator (marks as ended, stops accepting audio)
        if hasattr(self, '_streaming_request_generator'):
            self._streaming_request_generator.close_stream(stream_id)
        
        # State cleanup is now deferred - DO NOT delete from _state_pool here
        # The caller (e.g., cache_aware_pipeline_modal.py) should handle cleanup
        # after confirming all buffered frames have been processed

    async def process_streaming_batch(self):
        """
        Process the next batch of ready frames from streaming sessions.
        This is async to work with WebSocket handlers, but calls sync inference internally.
        Returns:
            tuple: (requests, outputs) where:
                - requests: list[Request] - The frames/feature buffers that were processed
                - outputs: list[TranscribeStepOutput] - Transcription outputs
            Returns ([], []) if no frames ready
        """
        if not hasattr(self, '_streaming_request_generator'):
            raise RuntimeError("Pipeline not configured for streaming.")
        
        try:
            # Try to get ready frames (non-blocking)
            requests = next(self._streaming_request_generator)
            # Process the batch (sync inference)
            # t0 = time.perf_counter()
            outputs = self.transcribe_step(requests)
            # t1 = time.perf_counter()
            # print(f"   🧹 Transcribe step took {t1 - t0} seconds")
            return requests, outputs
        except StopIteration:
            # All streams ended and consumed
            return [], []
        except Exception as e:
            # Check if it's NotEnoughDataException (can't import type here)
            if type(e).__name__ == 'NotEnoughDataException':
                # No frames ready yet
                return [], []
            # Re-raise other exceptions
            raise

    @abstractmethod
    def transcribe_step_for_frames(self, frames: list[Frame]) -> None:
        """Transcribe a step for frames"""
        pass

    @abstractmethod
    def transcribe_step_for_feature_buffers(self, fbuffers: list[FeatureBuffer]) -> None:
        """Transcribe a step for feature buffers"""
        pass

    @abstractmethod
    def get_sep(self) -> str:
        """Return the separator for the text postprocessor."""
        pass

    def translate_step(self, states: list[StreamingState], step_outputs: list[TranscribeStepOutput]) -> None:
        """
        Translate step
        Args:
            states (list[StreamingState]): List of StreamingState objects.
            step_outputs (list[TranscribeStepOutput]): List of TranscribeStepOutput objects.
        """
        src_langs, tgt_langs = [], []
        asr_transcripts, current_prefixes, previous_translations = [], [], []
        final_transcript_mask = []
        states_to_translate = []

        src_contexts, tgt_contexts = [], []
        for state, step_output in zip(states, step_outputs):
            if not state.options.enable_nmt:
                continue

            src_lang = state.options.source_language
            tgt_lang = state.options.target_language
            if not src_lang or not tgt_lang:
                raise ValueError("Source and target languages must be set when NMT is enabled")

            final = step_output.final_transcript
            partial = step_output.partial_transcript
            if not (final.strip() or partial.strip()):
                continue

            transcript = final or partial
            is_final = bool(final)
            prev_translation, prefix = state.previous_translation_info

            states_to_translate.append((state, step_output))
            src_langs.append(src_lang)
            tgt_langs.append(tgt_lang)
            asr_transcripts.append(transcript)
            current_prefixes.append(prefix)
            previous_translations.append(prev_translation)
            final_transcript_mask.append(is_final)

            src_context, tgt_context = state.previous_context
            src_contexts.append(src_context)
            tgt_contexts.append(tgt_context)

        if len(states_to_translate) == 0:
            return

        translations = self.nmt_model.translate(
            asr_transcripts, current_prefixes, src_langs, tgt_langs, src_contexts, tgt_contexts
        )
        new_prefixes = self.nmt_model.get_prefixes(asr_transcripts, translations, previous_translations)

        for (state, step_output), translation, new_prefix, prev_prefix, is_final in zip(
            states_to_translate, translations, new_prefixes, current_prefixes, final_transcript_mask
        ):
            if is_final:
                step_output.final_translation = translation
                step_output.partial_translation = ""
                state.cleanup_translation_info_after_eou()
                state.set_translation_context(step_output.final_transcript, translation)
                new_prefix = translation
            else:
                step_output.partial_translation = translation
                step_output.final_translation = ""
                state.set_translation_info(translation, new_prefix)

            lcp = os.path.commonprefix([prev_prefix, new_prefix])
            step_output.current_step_translation = new_prefix[len(lcp) :]

    def transcribe_step(self, requests: list[Request]) -> list[TranscribeStepOutput]:
        """
        Transcribe step
        Args:
            requests (list[Request]): List of Request objects.
        Returns:
            list[TranscribeStepOutput]: List of TranscribeStepOutput objects.
        """

        # Initialize the state if it is the first request for the stream
        states = []
        for request in requests:
            if request.is_first:
                self.init_state(request.stream_id, request.options)
            states.append(self.get_state(request.stream_id))

        # Perform the transcribe step for the frames or feature buffers
        if isinstance(requests[0], Frame):
            self.transcribe_step_for_frames(frames=requests)
        elif isinstance(requests[0], FeatureBuffer):
            self.transcribe_step_for_feature_buffers(fbuffers=requests)
        else:
            raise ValueError(f"Invalid request type: {type(requests[0])}")

        # Create current step output for each request
        outputs = []
        sep = self.get_sep()
        for request, state in zip(requests, states):
            step_output = TranscribeStepOutput.from_state(state=state, request=request, sep=sep)
            outputs.append(step_output)

        # Perform the translation step
        if self.nmt_enabled:
            self.translate_step(states=states, step_outputs=outputs)

        # Cleanup the states after the response is sent
        # If last request, delete state from the state pool to free memory
        for state, request in zip(states, requests):
            state.cleanup_after_response()
            if request.is_last:
                self.delete_state(request.stream_id)
        return outputs

    def copy_asr_model_attributes(self, asr_model: ASRInferenceWrapper) -> None:
        """
        Copy the attributes from the ASR model
        Args:
            asr_model (ASRInferenceWrapper): ASR model to copy the attributes from.
        """
        self.asr_model = asr_model
        self.tokenizer = asr_model.tokenizer
        self.device = asr_model.device
        self.supports_punctuation = asr_model.supports_punctuation()
        self.asr_supported_puncts = asr_model.supported_punctuation()
        self.leading_regex_pattern = get_leading_punctuation_regex_pattern(self.asr_supported_puncts)
        self.blank_id = asr_model.get_blank_id()
        self.vocabulary = asr_model.get_vocabulary()
        self.sep = asr_model.word_separator
        self.underscore_id = asr_model.underscore_id
        self.punctuation_ids = asr_model.punctuation_ids
        self.language_token_ids = asr_model.language_token_ids
        self.preprocessor, self.preprocessor_config = asr_model.create_preprocessor()
        self.subsampling_factor = asr_model.get_subsampling_factor()
        self.window_stride = asr_model.get_window_stride()
        self.model_stride_in_secs = asr_model.get_model_stride(in_secs=True)
        self.model_stride_in_milliseconds = asr_model.get_model_stride(in_milliseconds=True)

    def update_partial_transcript(
        self, requests: list[Request], tokenizer: TokenizerSpec, leading_regex_pattern: str
    ) -> None:
        """
        Update partial and current step transcripts from the state.
        Args:
            requests (list[Request]): List of Request objects.
            tokenizer (TokenizerSpec): Used to convert tokens into text
            leading_regex_pattern (str): Regex pattern for the punctuation marks.
        """
        word_separator = self.get_sep()
        for request in requests:
            state = self.get_state(request.stream_id)
            # state tokens represent all tokens accumulated since the EOU
            # incomplete segment tokens are the remaining tokens on the right side of the buffer after EOU
            all_tokens = state.tokens + state.incomplete_segment_tokens
            if len(all_tokens) > 0:
                pt_string = ids_to_text_without_stripping(all_tokens, tokenizer, word_separator)
                if leading_regex_pattern:
                    pt_string = re.sub(leading_regex_pattern, r'\1', pt_string)
                state.partial_transcript = pt_string
            else:
                state.partial_transcript = ""

            current_step_tokens = state.current_step_tokens
            if len(current_step_tokens) > 0:
                step_transcript = ids_to_text_without_stripping(current_step_tokens, tokenizer, word_separator)
                state.current_step_transcript = step_transcript
            else:
                state.current_step_transcript = ""

    def init_bpe_decoder(self) -> None:
        """Initialize the BPE decoder"""
        check_existance_of_required_attributes(
            self,
            [
                'vocabulary',
                'tokenizer',
                'confidence_aggregator',
                'asr_supported_puncts',
                'word_boundary_tolerance',
                'model_stride_in_secs',
            ],
        )

        self.bpe_decoder = BPEDecoder(
            vocabulary=self.vocabulary,
            tokenizer=self.tokenizer,
            confidence_aggregator=self.confidence_aggregator,
            asr_supported_puncts=self.asr_supported_puncts,
            word_boundary_tolerance=self.word_boundary_tolerance,
            token_duration_in_secs=self.model_stride_in_secs,
        )

    def init_text_processor(
        self,
        cfg: DictConfig,
        itn_model: AlignmentPreservingInverseNormalizer | None,
    ) -> None:
        """
        Initialize the text processor.
        Args:
            cfg: (DictConfig) Configuration parameters.
            itn_model: (AlignmentPreservingInverseNormalizer | None) Inverse Text Normalization model.
        """
        check_existance_of_required_attributes(
            self,
            [
                'asr_supported_puncts',
                'supports_punctuation',
                'confidence_aggregator',
                'sep',
            ],
        )

        self.text_processor = StreamingTextProcessor(
            itn_cfg=cfg.itn,
            itn_model=itn_model,
            asr_supported_puncts=self.asr_supported_puncts,
            asr_supports_punctuation=self.supports_punctuation,
            confidence_aggregator=self.confidence_aggregator,
            sep=self.sep,
            enable_pnc=cfg.enable_pnc,
            enable_itn=cfg.enable_itn,
        )

    def init_nmt_model(self, nmt_model: LLMTranslator | None) -> None:
        """
        Initialize the Translation model.
        Args:
            nmt_model: (LLMTranslator | None) LLM based translation model.
        """
        self.nmt_model = nmt_model
        self.nmt_enabled = nmt_model is not None

    def init_bufferer_for_buffered_streaming(self) -> None:
        """Initialize the bufferer."""
        check_existance_of_required_attributes(
            self,
            [
                'request_type',
                'sample_rate',
                'buffer_size_in_secs',
                'preprocessor_config',
                'device',
            ],
        )

        if self.request_type is RequestType.FEATURE_BUFFER:
            # Feature buffering: It will be used when the input is feature buffers
            self.bufferer = BatchedFeatureBufferer(
                sample_rate=self.sample_rate,
                buffer_size_in_secs=self.buffer_size_in_secs,
                preprocessor_cfg=self.preprocessor_config,
                device=self.device,
            )
        elif self.request_type is RequestType.FRAME:
            # Audio buffering: It will be used when the input is audio frames
            self.bufferer = BatchedAudioBufferer(
                sample_rate=self.sample_rate, buffer_size_in_secs=self.buffer_size_in_secs
            )
        else:
            raise ValueError(f"Unknown request type: {self.request_type}")

    def init_bufferer_for_cache_aware_streaming(self) -> None:
        """Initialize the bufferer for cache-aware streaming."""
        check_existance_of_required_attributes(
            self,
            [
                'num_slots',
                'use_feat_cache',
                'chunk_size_in_secs',
                'buffer_size_in_secs',
                'sample_rate',
                'preprocessor_config',
                'device',
            ],
        )

        if self.use_feat_cache:
            # Only calculate mel-spec features for last chunk
            chunk_size_for_feature_buffer = self.chunk_size_in_secs
        else:
            # Calculate mel-spec features for the whole buffer
            chunk_size_for_feature_buffer = self.buffer_size_in_secs

        self.bufferer = BatchedCacheFeatureBufferer(
            num_slots=self.num_slots,
            sample_rate=self.sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=chunk_size_for_feature_buffer,
            preprocessor_cfg=self.preprocessor_config,
            device=self.device,
        )

    def init_context_manager(self) -> None:
        """Initialize the context manager."""
        check_existance_of_required_attributes(self, ['asr_model', 'num_slots', 'use_cache'])
        self.context_manager = CacheAwareContextManager(
            cache_aware_model=self.asr_model, num_slots=self.num_slots, use_cache=self.use_cache
        )


class CacheAwareRNNTStreamingPipeline(BaseStreamingPipeline):
    """Cache Aware RNNT pipeline."""

    def __init__(
        self,
        cfg: DictConfig,
        asr_model: CacheAwareRNNTInferenceWrapper,
        itn_model: AlignmentPreservingInverseNormalizer | None = None,
        nmt_model: LLMTranslator | None = None,
    ):
        """
        Initialize the CacheAwareRNNTStreamingPipeline.
        Args:
            cfg: (DictConfig) Configuration parameters.
            asr_model: (CacheAwareRNNTInferenceWrapper) ASR model.
            itn_model: (AlignmentPreservingInverseNormalizer | None) Inverse Text Normalization model.
            nmt_model: (LLMTranslator | None) LLM based translation model.
        """
        self.copy_asr_model_attributes(asr_model)
        self.init_parameters(cfg)
        self.init_context_manager()
        self.init_bufferer_for_cache_aware_streaming()
        self.conf_func, self.confidence_aggregator = get_confidence_utils(cfg.confidence)
        self.init_bpe_decoder()
        self.init_greedy_rnnt_decoder()
        self.init_endpointer()
        self.init_text_processor(cfg, itn_model)
        self.init_nmt_model(nmt_model)
        super().__init__()

    def init_parameters(self, cfg: DictConfig) -> None:
        """
        Initialize the parameters.
        Args:
            cfg: (DictConfig) Configuration parameters.
        """
        if cfg.streaming.att_context_size is not None:
            self.asr_model.set_default_att_context_size(att_context_size=cfg.streaming.att_context_size)

        self.sample_rate = cfg.streaming.sample_rate
        self.asr_output_granularity = cfg.asr_output_granularity
        self.pre_encode_cache_size = self.asr_model.get_pre_encode_cache_size()
        self.model_chunk_size = self.asr_model.get_chunk_size()
        if isinstance(self.model_chunk_size, list):
            self.model_chunk_size = self.model_chunk_size[1]

        self.use_cache = cfg.streaming.use_cache
        self.use_feat_cache = cfg.streaming.use_feat_cache

        if cfg.streaming.get("chunk_size_in_secs", None) is not None:
            self.chunk_size_in_secs = cfg.streaming.chunk_size_in_secs
            self.tokens_per_frame = math.ceil(
                np.trunc(self.chunk_size_in_secs / self.window_stride) / self.subsampling_factor
            )
            # overwrite the encoder streaming params with proper shift size for cache aware streaming
            self.asr_model.setup_streaming_params(
                chunk_size=self.model_chunk_size // self.subsampling_factor, shift_size=self.tokens_per_frame
            )
        else:
            self.chunk_size_in_secs = self.model_chunk_size * self.window_stride
            self.tokens_per_frame = math.ceil(self.model_chunk_size / self.subsampling_factor)

        if isinstance(self.pre_encode_cache_size, list):
            self.pre_encode_cache_size = self.pre_encode_cache_size[1]
        self.pre_encode_cache_size_in_secs = self.pre_encode_cache_size * self.window_stride

        # Context Manager
        self.batch_size = cfg.streaming.batch_size
        self.num_slots = cfg.streaming.num_slots
        if self.num_slots < self.batch_size:
            raise ValueError(
                f"Number of slots in the context manager must be >= batch_size: {self.num_slots} < {self.batch_size}"
            )
        model_chunk_size_in_secs = self.model_chunk_size * self.window_stride

        if self.use_cache:
            # if using cache, we need to pad some samples for pre_encode
            self.buffer_size_in_secs = self.pre_encode_cache_size_in_secs + model_chunk_size_in_secs
            self.drop_left_context = None
            self.valid_out_len = None
        else:
            # if not using cache, we need to keep left context in buffer, but no extra padding in pre_encode
            left_context_size = self.asr_model.get_att_context_size()[0]
            if left_context_size < 0:
                raise ValueError(f"Left context size should not be a negative value: {left_context_size}")
            self.buffer_size_in_secs = (
                model_chunk_size_in_secs + left_context_size * self.subsampling_factor * self.window_stride
            )
            self.drop_left_context = left_context_size
            self.valid_out_len = self.tokens_per_frame

        self.stop_history_eou_in_milliseconds = cfg.endpointing.stop_history_eou
        self.residue_tokens_at_end = cfg.endpointing.residue_tokens_at_end
        self.word_boundary_tolerance = cfg.streaming.word_boundary_tolerance
        self.return_tail_result = cfg.return_tail_result

        self.request_type = RequestType.from_str(cfg.streaming.request_type)
        if self.request_type is not RequestType.FRAME:
            raise ValueError(f"Request type {self.request_type} is not supported for cache-aware streaming.")

    def init_greedy_rnnt_decoder(self) -> None:
        """Initialize the RNNT decoder."""
        check_existance_of_required_attributes(self, ['vocabulary', 'conf_func'])
        self.greedy_rnnt_decoder = RNNTGreedyDecoder(vocabulary=self.vocabulary, conf_func=self.conf_func)

    def init_endpointer(self) -> None:
        """Initialize the endpointer."""
        check_existance_of_required_attributes(
            self,
            [
                'vocabulary',
                'model_stride_in_milliseconds',
                'stop_history_eou_in_milliseconds',
                'residue_tokens_at_end',
            ],
        )

        self.endpointer = RNNTGreedyEndpointing(
            vocabulary=self.vocabulary,
            ms_per_timestep=self.model_stride_in_milliseconds,
            stop_history_eou=self.stop_history_eou_in_milliseconds,
            residue_tokens_at_end=self.residue_tokens_at_end,
        )

    def create_state(self, options: ASRRequestOptions) -> CacheAwareRNNTStreamingState:
        """
        Create new empty state.
        Args:
            options: (ASRRequestOptions) Request options for particular stream.
        Returns:
            (CacheAwareRNNTStreamingState) New empty state.
        """
        state = CacheAwareRNNTStreamingState()
        state.set_global_offset(0)
        new_options = options.augment_with_defaults(
            default_enable_itn=self.text_processor.is_itn_enabled(),
            default_enable_pnc=self.text_processor.is_pnc_enabled(),
            default_enable_nmt=self.nmt_enabled,
            default_source_language=self.nmt_model.source_language if self.nmt_enabled else None,
            default_target_language=self.nmt_model.target_language if self.nmt_enabled else None,
            default_stop_history_eou=self.stop_history_eou_in_milliseconds,
            default_asr_output_granularity=self.asr_output_granularity,
        )

        eou_label_buffer_size = 0
        if new_options.stop_history_eou > 0:
            eou_label_buffer_size = millisecond_to_frames(
                new_options.stop_history_eou, math.ceil(self.model_stride_in_milliseconds)
            )
            eou_label_buffer_size += self.residue_tokens_at_end
        state.setup_label_buffer(eou_label_buffer_size, self.blank_id)
        state.set_previous_hypothesis(None)
        state.set_options(new_options)
        return state

    def get_sep(self) -> str:
        """Return the separator for the text processor."""
        return self.sep

    def preprocess(self, buffers: list[Tensor], right_paddings: list[int] | None = None) -> tuple[Tensor, Tensor]:
        """
        Preprocess the feature buffers by stacking them and computing the lengths
        Args:
            buffers: (list[Tensor]) List of feature buffers.
            right_paddings: (list[int] | None) List of right paddings.
        Returns:
            (tuple[Tensor, Tensor]) Processed feature buffers and their lengths.
        """
        feature_buffers = [f_buffer.unsqueeze_(0) for f_buffer in buffers]
        feature_buffer_lens = torch.tensor([f_buffer.shape[2] for f_buffer in feature_buffers], device=self.device)
        if right_paddings is not None:
            right_paddings = torch.tensor(right_paddings, device=feature_buffer_lens.device)
            feature_buffer_lens = feature_buffer_lens - right_paddings
        feature_buffers = torch.cat(feature_buffers).to(self.device)
        return feature_buffers, feature_buffer_lens

    def run_greedy_decoder(self, state: CacheAwareRNNTStreamingState, frame: Frame, hyp: Hypothesis) -> bool:
        """
        Run the greedy RNNT decoder on the hypothesis and update the state
        Args:
            state: (CacheAwareRNNTStreamingState) The state of the stream
            frame: (Frame) The current frame
            hyp: (Hypothesis) The hypothesis of the current frame
        Returns:
            (bool) Whether EOU is detected.
        """
        eou_detected = frame.is_last
        cur_output, cur_labels, new_offset = self.greedy_rnnt_decoder(
            global_timestamps=hyp.timestamp,
            tokens=hyp.y_sequence,
            length=self.tokens_per_frame,
            offset=state.offset,
        )
        state.set_offset(new_offset)

        # cur labels contains blank tokens as well, it is needed for EOU detection
        state.update_label_buffer(cur_labels)

        if not eou_detected:
            emissions = state.get_label_buffer()
            pivot_point = len(emissions) - 1
            eou_detected, _ = self.endpointer.detect_eou_near_pivot(
                emissions, pivot_point, stop_history_eou=state.options.stop_history_eou
            )

        state.update_state(cur_output, eou_detected=eou_detected)
        return eou_detected

    def cache_aware_transcribe_step(
        self,
        frames: list[Frame],
        features: list[Tensor],
        right_paddings: list[int],
        ready_state_ids: set,
        keep_all_outputs: bool = False,
    ) -> None:
        """
        Cache Aware Transcribe Step
        It receives a list of frames and features and do the following:

        1. Preprocess the features by stacking them and computing the lengths
        2. Collecting previous hypotheses for stateful decoding
        3. Get the context and mapping from the context manager for cache aware streaming
        4. Perform a streaming step with the ASR model
        5. Update the cache and reset the cache slots for the streams that has ended
        6. Update the previous hypothesis and reset the previous hypothesis for the streams that has ended
        7. Perform greedy RNNT decoding to get the best hypothesis and update the states
        8. Update the ready states to indicate that the state is ready for text post-processing
        Args:
            frames: (list[Frame]) List of frames to transcribe.
            features: (list[Tensor]) List of feature buffers.
            right_paddings: (list[int] | None) List of right paddings.
            ready_state_ids: (set) Set of ready state IDs.
            keep_all_outputs: (bool) Whether to keep all outputs or not.
        """

        feature_buffers, feature_buffer_lens = self.preprocess(features, right_paddings)
        states, stream_ids, eos_flags = [], [], []
        for frame in frames:
            states.append(self.get_state(frame.stream_id))
            stream_ids.append(frame.stream_id)
            eos_flags.append(frame.is_last)

        previous_hypotheses = [state.get_previous_hypothesis() for state in states]
        context, mapping = self.context_manager.get_context(stream_ids)

        drop_extra_pre_encoded = 0 if not self.use_cache else self.asr_model.drop_extra_pre_encoded
        best_hyp, new_context = self.asr_model.stream_step(
            processed_signal=feature_buffers,
            processed_signal_length=feature_buffer_lens,
            context=context,
            previous_hypotheses=previous_hypotheses,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            keep_all_outputs=keep_all_outputs,
            drop_left_context=self.drop_left_context,
            valid_out_len=self.valid_out_len,
        )

        # update the cache and reset the cache slots for the streams that has ended
        self.context_manager.update_cache(stream_ids, new_context, mapping)
        self.context_manager.reset_slots(stream_ids, eos_flags)

        # update the previous hypothesis and reset the previous hypothesis for the streams that has ended
        for state, hyp, eos in zip(states, best_hyp, eos_flags):
            if eos:
                state.reset_previous_hypothesis()
            else:
                state.set_previous_hypothesis(hyp)

        # run greedy decoder for each frame-state-hypothesis tuple
        for frame, state, hyp in zip(frames, states, best_hyp):
            eou_detected = self.run_greedy_decoder(state, frame, hyp)
            if eou_detected:
                self.bpe_decoder.decode_bpe_tokens(state)
                state.cleanup_after_eou()
                ready_state_ids.add(frame.stream_id)

    def transcribe_step_for_feature_buffers(self, fbuffers: list[FeatureBuffer]) -> None:
        """Transcribe a step for feature buffers"""
        raise NotImplementedError("Feature buffer type is not supported for cache aware streaming.")

    def transcribe_step_for_frames(self, frames: list[Frame]) -> None:
        """
        Transcribes the frames in a streaming manner.
        After detecting EOU, it updates the state and run text processor.
        If there are multiple streams, it waits until all states are ready to run text processor.
        Args:
            frames: (list[Frame]) List of frames to transcribe.
        """

        all_fbuffers, right_paddings = self.bufferer.update(frames)
        ready_state_ids = set()

        # streams that contains multiple frames
        if len(all_fbuffers) > 0:
            final_frames, final_fbuffers = [], []
            nonfinal_frames, nonfinal_fbuffers = [], []
            final_right_paddings = []
            for jdx, bfeature in enumerate(all_fbuffers):
                bframe = frames[jdx]

                if bframe.is_last:
                    final_frames.append(bframe)
                    final_fbuffers.append(bfeature)
                    final_right_paddings.append(right_paddings[jdx])
                else:
                    nonfinal_frames.append(bframe)
                    nonfinal_fbuffers.append(bfeature)

            if len(nonfinal_frames) > 0:
                self.cache_aware_transcribe_step(
                    nonfinal_frames, nonfinal_fbuffers, None, ready_state_ids, keep_all_outputs=False
                )

            if len(final_frames) > 0:
                self.cache_aware_transcribe_step(
                    final_frames, final_fbuffers, final_right_paddings, ready_state_ids, keep_all_outputs=True
                )

        # post-process the ready states
        if len(ready_state_ids) > 0:
            self.text_processor.process([self.get_state(stream_id) for stream_id in ready_state_ids])
            ready_state_ids.clear()

        self.update_partial_transcript(frames, self.tokenizer, self.leading_regex_pattern)


    def init_streaming_request_generator(self):
        """
        Initialize the streaming request generator for real-time audio.
        This should be called before using streaming methods.
        """
        from .multi_stream_utils import StreamingBatchedRequestStreamer
        
        self._streaming_request_generator = StreamingBatchedRequestStreamer(
            sample_rate=self.sample_rate,
            frame_size_in_secs=self.chunk_size_in_secs,
            batch_size=self.batch_size,
            n_frames_per_stream=1,  # Process one frame at a time for cache-aware
            request_type=self.request_type,
            preprocessor=None,
            buffer_size_in_secs=None,
            device=None,
            pad_last_frame=True,
        )


class CacheAwareStreamingPipelineBuilder(BaseBuilder):
    """
    Cache Aware Streaming Pipeline Builder class.
    Builds the cache aware CTC/RNNT pipelines.
    """

    @classmethod
    def build(cls, cfg: DictConfig) -> CacheAwareCTCPipeline | CacheAwareRNNTStreamingPipeline:
        """
        Build the cache aware streaming pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns CacheAwareCTCStreamingPipeline [Not implemented] or CacheAwareRNNTStreamingPipeline object
        """
        asr_decoding_type = ASRDecodingType.from_str(cfg.asr_decoding_type)

        if asr_decoding_type is ASRDecodingType.RNNT:
            return cls.build_cache_aware_rnnt_pipeline(cfg)
        elif asr_decoding_type is ASRDecodingType.CTC:
            raise ValueError("Cache aware CTC pipeline is not implemented for streaming.")

        raise ValueError("Invalid asr decoding type for cache aware streaming. Need to be one of ['CTC', 'RNNT']")

    @classmethod
    def get_rnnt_decoding_cfg(cls, cfg: DictConfig) -> RNNTDecodingConfig:
        """
        Get the decoding config for the RNNT pipeline.
        Returns:
            (RNNTDecodingConfig) Decoding config
        """
        base_cfg_structured = OmegaConf.structured(RNNTDecodingConfig)
        base_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg_structured))
        decoding_cfg = OmegaConf.merge(base_cfg, cfg.asr.decoding)
        return decoding_cfg

    @classmethod
    def get_ctc_decoding_cfg(cls) -> CTCDecodingConfig:
        """
        Get the decoding config for the CTC pipeline.
        Returns:
            (CTCDecodingConfig) Decoding config
        """
        decoding_cfg = CTCDecodingConfig()
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = False
        return decoding_cfg

    @classmethod
    def build_cache_aware_rnnt_pipeline(cls, cfg: DictConfig) -> CacheAwareRNNTStreamingPipeline:
        """
        Build the cache aware RNNT streaming pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns CacheAwareRNNTStreamingPipeline object
        """
        # building ASR model
        decoding_cfg = cls.get_rnnt_decoding_cfg(cfg)
        asr_model = cls._build_asr(cfg, decoding_cfg)

        # building ITN model
        itn_model = cls._build_itn(cfg, input_is_lower_cased=True)

        # building NMT model
        nmt_model = cls._build_nmt(cfg)

        # building cache aware RNNT pipeline
        ca_rnnt_pipeline = CacheAwareRNNTStreamingPipeline(cfg, asr_model, itn_model, nmt_model)
        logging.info(f"`{type(ca_rnnt_pipeline).__name__}` pipeline loaded")
        return ca_rnnt_pipeline

    @classmethod
    def build_cache_aware_ctc_pipeline(cls, cfg: DictConfig) -> CacheAwareCTCPipeline:
        """
        Build the cache aware CTC streaming pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns CacheAwareCTCPipeline object
        """
        # building ASR model
        decoding_cfg = cls.get_ctc_decoding_cfg()
        asr_model = cls._build_asr(cfg, decoding_cfg)

        # building ITN model
        itn_model = cls._build_itn(cfg, input_is_lower_cased=True)

        # building NMT model
        nmt_model = cls._build_nmt(cfg)

        # building cache aware CTC pipeline
        ca_ctc_pipeline = CacheAwareCTCPipeline(cfg, asr_model, itn_model, nmt_model)
        logging.info(f"`{type(ca_ctc_pipeline).__name__}` pipeline loaded")
        return ca_ctc_pipeline


class StreamingPipelineBuilder:
    """Router for building the streaming pipeline based on the pipeline type."""

    @staticmethod
    def set_matmul_precision(matmul_precision: str) -> None:
        """
        Set the matmul precision.
        Args:
            matmul_precision: (str) Matmul precision: highest, high, medium
        """
        choices = ["highest", "high", "medium"]
        matmul_precision = matmul_precision.lower()
        if matmul_precision not in choices:
            raise ValueError(f"Invalid matmul precision: {matmul_precision}. Need to be one of {choices}")
        torch.set_float32_matmul_precision(matmul_precision)
        logging.info(f"Using matmul precision: {matmul_precision}")

    @staticmethod
    def set_log_level(log_level: int) -> None:
        """
        Set the logging level.
        Args:
            log_level: (int) Logging level: 0 (NOTSET), 10 (DEBUG), 20 (INFO), 30 (WARNING), 40 (ERROR), 50 (CRITICAL)
        """
        choices = [0, 10, 20, 30, 40, 50]
        if log_level not in choices:
            raise ValueError(f"Invalid log level: {log_level}. Need to be one of {choices}")
        logging.setLevel(log_level)

    @staticmethod
    def build_pipeline(cfg: DictConfig) -> Any:
        """
        Build the pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns Pipeline object
        """
        StreamingPipelineBuilder.set_log_level(cfg.log_level)
        StreamingPipelineBuilder.set_matmul_precision(cfg.matmul_precision)
        pipeline_type = PipelineType.from_str(cfg.pipeline_type)
        if pipeline_type is PipelineType.BUFFERED:
            # builder = BufferedPipelineBuilder
            raise ValueError("Buffered pipeline is not implemented for streaming.")
        elif pipeline_type is PipelineType.CACHE_AWARE:
            builder = CacheAwareStreamingPipelineBuilder
        else:
            raise ValueError(f"Invalid streaming pipeline type: {cfg.pipeline_type}")

        return builder.build(cfg)
