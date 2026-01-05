// Audio Worklet Processor for streaming audio to WebSocket
//
// This processor:
// 1. Receives audio at the AudioContext sample rate (16kHz)
// 2. Buffers samples until we have 80ms worth (1280 samples at 16kHz)
// 3. Converts Float32 audio to Int16 PCM format
// 4. Sends chunks to main thread for WebSocket transmission
//
// Chunk size: 80ms = 1280 samples at 16kHz = 2560 bytes (Int16)
// This matches the backend's expected chunk size for optimal real-time performance

/* global AudioWorkletProcessor, registerProcessor */

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // 80ms chunks at 16kHz = 1280 samples
        // This matches the backend's expected chunk size for optimal real-time performance
        this.targetChunkSize = 1280;
        this.buffer = [];
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        
        if (input.length > 0) {
            const channelData = input[0]; // Get first channel (mono)
            
            // Add samples to buffer
            for (let i = 0; i < channelData.length; i++) {
                this.buffer.push(channelData[i]);
            }
            
            // When we have enough samples, send them
            while (this.buffer.length >= this.targetChunkSize) {
                // Extract chunk
                const chunk = this.buffer.splice(0, this.targetChunkSize);
                
                // Convert Float32Array to Int16Array
                const int16Data = new Int16Array(chunk.length);
                for (let i = 0; i < chunk.length; i++) {
                    // Clamp to [-1, 1] and convert to 16-bit integer
                    const sample = Math.max(-1, Math.min(1, chunk[i]));
                    int16Data[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                }
                
                // Send to main thread
                this.port.postMessage(int16Data.buffer, [int16Data.buffer]);
            }
        }
        
        return true; // Keep processor alive
    }
}

registerProcessor('audio-processor', AudioProcessor);

