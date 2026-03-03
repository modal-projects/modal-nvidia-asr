#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.nvidia.tts import NvidiaTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}

async def run_bot(room_url, token):
    logger.info(f"Starting bot: {room_url} - {token}")

    # Set up Daily transport with video/audio parameters
    transport = DailyTransport(
        room_url,
        token,
        "NemotronBot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    stt = NvidiaSTTService(
        api_key="OPEN_WEIGHTS",
        server = "modal-labs-shababo-dev--nim-riva-asr-rivaasr.us-west.modal.direct",
        model_function_map = {
            "model_name": "parakeet-0-6b-ctc-en-us"
        }
    )

    llm = NvidiaLLMService(
        api_key="OPEN_WEIGHTS",
        base_url="https://modal-labs-shababo-dev--nim-nemotron-nano-3-nemotronnano3.us-west.modal.direct/v1",
        model="nvidia/nemotron-3-nano",
    )

    tts = NvidiaTTSService(
        api_key="OPEN_WEIGHTS",
        server="modal-labs-shababo-dev--nim-riva-tts-rivatts.us-west.modal.direct"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way, but be concise and use two to three short sentences to answer the question.",
        },
    ]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner()

    await runner.run(task)
