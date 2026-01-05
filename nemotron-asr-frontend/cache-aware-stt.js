// WebSocket connection
let ws = null;
let audioContext = null;
let audioWorklet = null;
let micStream = null;
let isRecording = false;
let streamId = null; // Track stream ID from server

// UI elements
const recordBtn = document.getElementById('recordBtn');
// const statusDiv = document.getElementById('status');
const transcriptDiv = document.getElementById('transcript');
const audioTimestampSpan = document.getElementById('audio-timestamp');
// const inferenceTimeSpan = document.getElementById('inference-time');
const stepSpan = document.getElementById('step');
// const isFinalSpan = document.getElementById('is-final');

// Transcription state
let fullTranscript = '';
let currentPartial = '';
let stepCount = 0; // Track step count locally

// Get WebSocket URL from window variable (injected by server)
const WS_BASE_URL = window.WS_BASE_URL || `ws://${window.location.host}/ws`;

// Initialize WebSocket connection
function connectWebSocket() {
    return new Promise((resolve, reject) => {
        console.log('Connecting to WebSocket...');
        ws = new WebSocket(WS_BASE_URL);
        ws.binaryType = 'arraybuffer'; // Receive binary data as ArrayBuffer
        
        ws.onopen = () => {
            console.log('WebSocket connected');
            resolve();
        };
        
        ws.onmessage = async (event) => {
            try {
                console.log('Raw message received, type:', typeof event.data, 
                    'instanceof ArrayBuffer:', event.data instanceof ArrayBuffer,
                    'instanceof Blob:', event.data instanceof Blob,
                    'constructor:', event.data.constructor.name);
                
                let data;
                
                // Backend sends JSON text messages for raw bytes clients
                if (typeof event.data === 'string') {
                    // Check for marker echo
                    if (event.data === 'END' || event.data === 'MARKER') {
                        console.log('✅ Received text Marker echo, closing connection');
                        stopRecording();
                        return;
                    }
                    
                    // Parse JSON message
                    try {
                        data = JSON.parse(event.data);
                    } catch (e) {
                        console.warn('Received non-JSON text message:', event.data);
                        return;
                    }
                } else if (event.data instanceof ArrayBuffer) {
                    // Try to decode as text/JSON
                    const text = new TextDecoder().decode(event.data);
                    console.log('Decoded ArrayBuffer to text:', text.substring(0, 100));
                    try {
                        data = JSON.parse(text);
                    } catch (e) {
                        console.error('ArrayBuffer is not valid JSON:', text.substring(0, 100));
                        return;
                    }
                } else if (event.data instanceof Blob) {
                    // Convert Blob to text
                    const arrayBuffer = await event.data.arrayBuffer();
                    const text = new TextDecoder().decode(arrayBuffer);
                    console.log('Decoded Blob to text:', text.substring(0, 100));
                    try {
                        data = JSON.parse(text);
                    } catch (e) {
                        console.error('Blob is not valid JSON:', text.substring(0, 100));
                        return;
                    }
                } else {
                    console.error('Unexpected data type (expected text/JSON):', typeof event.data, event.data.constructor.name);
                    return;
                }
                
                console.log('Decoded message:', data);
                
                // Handle different message types
                if (data.type === 'Ready') {
                    streamId = data.id;
                    console.log('✅ Received Ready message, stream_id:', streamId);
                } else if (data.type === 'Marker') {
                    console.log('✅ Received Marker echo, closing connection');
                    stopRecording();
                } else {
                    // It's a transcription result (no 'type' field)
                    console.log('📝 Transcription result:', data);
                    handleTranscription(data);
                }
            } catch (e) {
                console.error('❌ Error parsing message:', e);
                console.error('Event data:', event.data);
            }
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            reject(error);
        };
        
        ws.onclose = () => {
            console.log('WebSocket closed');
            
            if (isRecording) {
                console.log('WebSocket closed while recording, stopping...');
                stopRecording();
                alert('Connection lost. Recording stopped.');
            }
            
            ws = null;
        };
    });
}

// Disconnect WebSocket
function disconnectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        console.log('Closing WebSocket...');
        ws.close();
        ws = null;
    }
}

// Handle incoming transcription
function handleTranscription(result) {
    console.log('📝 handleTranscription called with:', result);
    console.log('   is_final:', result.is_final, 'text:', result.text);
    
    // Increment and update step count
    stepCount++;
    stepSpan.textContent = stepCount;
    
    // Update metadata
    if (result.audio_timestamp !== undefined) {
        audioTimestampSpan.textContent = `${result.audio_timestamp.toFixed(2)}s`;
    }
    // inferenceTimeSpan.textContent = `${(result.inference_time * 1000).toFixed(0)}ms`;
    // isFinalSpan.textContent = result.is_final ? '✓ Final' : '⋯ Partial';
    
    // Update transcript based on whether it's final
    if (result.is_final) {
        // Final transcript: use the accumulated interim transcript (which has capitalization)
        // instead of the backend's final_transcript (which doesn't have capitalization)
        if (currentPartial && currentPartial.trim()) {
            fullTranscript += currentPartial; 
            console.log('✅ Final - using accumulated interim text:', currentPartial);
            console.log('   (Ignoring backend final text which lacks capitalization)');
            console.log('   Full transcript now:', fullTranscript);
        } else {
            // Fallback to backend's text if we somehow don't have a partial
            const finalText = result.segment_text || result.text;
            if (finalText && finalText.trim()) {
                fullTranscript += finalText;
                console.log('✅ Final text added (fallback):', finalText);
            } else {
                console.log('⚠️  Final result but no text (text:', result.text, 'segment_text:', result.segment_text, 'currentPartial:', currentPartial, ')');
            }
        }
        currentPartial = '';  // Clear partial for next utterance
    } else {
        // Partial/incremental text while speaking
        // Concatenate/accumulate each partial transcript
        if (result.text !== undefined && result.text !== null && result.text !== '') {
            currentPartial += result.text;
            console.log('✏️  Partial text accumulated:', currentPartial);
        } else {
            // Empty partial - don't clear accumulated text, just log
            console.log('⚠️  Partial result with empty text (keeping accumulated text)');
        }
    }
    
    // Always display both final and partial
    // Show final in black, partial in gray
    let displayHtml = '';
    if (fullTranscript) {
        displayHtml += `<span class="final">${escapeHtml(fullTranscript)}</span>`;
    }
    if (currentPartial) {
        displayHtml += `<span class="partial">${escapeHtml(currentPartial)}</span>`;
    }
    
    console.log('   Display HTML:', displayHtml ? 'has content' : 'empty');
    console.log('   fullTranscript:', fullTranscript.length, 'chars');
    console.log('   currentPartial:', currentPartial.length, 'chars');
    
    if (displayHtml) {
        transcriptDiv.innerHTML = displayHtml;
    } else {
        transcriptDiv.innerHTML = '<span class="partial">Listening...</span>';
    }
    
    // Scroll to bottom smoothly
    const container = transcriptDiv.parentElement;
    container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
    });
}

// Start recording
async function startRecording() {
    try {
        console.log('Starting recording...');
        
        console.log('✅ Using raw bytes + JSON protocol (no msgpack needed)');
        
        // Disable button during setup
        recordBtn.disabled = true;
        recordBtn.textContent = 'Connecting...';
        
        // Connect to WebSocket first
        await connectWebSocket();
        console.log('WebSocket connected, setting up audio...');
        
        // Request microphone access
        micStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        });
        
        // Create audio context with 16kHz sample rate
        audioContext = new AudioContext({ sampleRate: 16000 });
        
        // Log actual sample rate (browser may not honor requested rate)
        console.log(`🎧 AudioContext created with sample rate: ${audioContext.sampleRate} Hz`);
        
        // Warn if browser didn't honor our 16kHz request
        if (audioContext.sampleRate !== 16000) {
            console.warn(`⚠️  Browser is using ${audioContext.sampleRate} Hz instead of requested 16000 Hz`);
            console.warn('   Audio will be resampled by the browser, which may affect quality');
        }
        
        // Verify the mic stream settings
        const audioTrack = micStream.getAudioTracks()[0];
        const settings = audioTrack.getSettings();
        console.log('🎤 Microphone settings:', settings);
        
        // Create media stream source
        const source = audioContext.createMediaStreamSource(micStream);
        
        // Create audio worklet for processing
        await audioContext.audioWorklet.addModule('/static/audio-processor.js');
        audioWorklet = new AudioWorkletNode(audioContext, 'audio-processor');
        
        console.log('AudioWorklet created and ready');
        
        // Handle processed audio
        let audioChunkCount = 0;
        let loggedFirstChunk = false;
        audioWorklet.port.onmessage = (event) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                try {
                    // event.data is an ArrayBuffer of Int16 PCM data
                    // Send it directly as raw bytes - backend will handle it
                    
                    // Log first chunk in detail
                    if (!loggedFirstChunk) {
                        console.log('🔍 FIRST AUDIO CHUNK DETAILS:');
                        console.log('   ArrayBuffer size:', event.data.byteLength, 'bytes');
                        console.log('   Expected samples:', event.data.byteLength / 2, '(int16 = 2 bytes/sample)');
                        console.log('   Format: Raw PCM Int16 bytes');
                        console.log('   Backend will decode as raw audio (no msgpack needed)');
                        loggedFirstChunk = true;
                    }
                    
                    // Send raw ArrayBuffer directly - backend accepts raw PCM bytes!
                    ws.send(event.data);
                    
                    audioChunkCount++;
                    if (audioChunkCount % 50 === 0) {
                        console.log(`🎤 Sent ${audioChunkCount} audio chunks (${event.data.byteLength} bytes each)`);
                    }
                } catch (e) {
                    console.error('❌ Error sending audio message:', e);
                    console.error('   Event data type:', typeof event.data);
                    console.error('   Event data:', event.data);
                }
            }
        };
        
        // Connect nodes
        source.connect(audioWorklet);
        audioWorklet.connect(audioContext.destination);
        
        isRecording = true;
        recordBtn.textContent = 'Stop Recording';
        recordBtn.classList.add('recording');
        recordBtn.disabled = false;
        
        // Clear transcript and reset state
        fullTranscript = '';
        currentPartial = '';
        streamId = null;
        stepCount = 0;
        stepSpan.textContent = '0';
        audioTimestampSpan.textContent = '0.00s';
        transcriptDiv.innerHTML = '<span class="partial">Listening...</span>';
        
        console.log('Recording started');
        
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Error starting recording: ' + error.message);
        stopRecording();
    }
}

// Stop recording
function stopRecording() {
    console.log('Stopping recording...');
    
    isRecording = false;
    
    // Send Marker message to signal end of stream
    if (ws && ws.readyState === WebSocket.OPEN) {
        try {
            console.log('🔍 MARKER MESSAGE:');
            console.log('   Sending simple text marker: "END"');
            
            // Send simple text message - backend will recognize "END" or "MARKER"
            ws.send("END");
            
            console.log('✅ Marker message sent!');
        } catch (e) {
            console.error('❌ Error sending Marker message:', e);
        }
    }
    
    if (audioWorklet) {
        audioWorklet.disconnect();
        audioWorklet = null;
    }
    
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    
    if (micStream) {
        micStream.getTracks().forEach(track => track.stop());
        micStream = null;
    }
    
    // Disconnect WebSocket (after a brief delay to allow Marker to be sent)
    setTimeout(() => {
        disconnectWebSocket();
    }, 100);
    
    recordBtn.textContent = 'Start Recording';
    recordBtn.classList.remove('recording');
    recordBtn.disabled = false;
    
    console.log('Recording stopped and disconnected');
}

// Update status display
// function updateStatus(status) {
//     statusDiv.className = status;
    
//     switch(status) {
//         case 'idle':
//             statusDiv.textContent = '⚪ Ready - Click to start';
//             break;
//         case 'connecting':
//             statusDiv.textContent = '🔄 Connecting...';
//             break;
//         case 'connected':
//             statusDiv.textContent = '✓ Connected - Starting recording';
//             break;
//         case 'recording':
//             statusDiv.textContent = '🔴 Recording - Speak now';
//             break;
//         case 'error':
//             statusDiv.textContent = '⚠ Connection error';
//             break;
//     }
// }

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Button click handler
recordBtn.addEventListener('click', () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

// Initialize on page load
window.addEventListener('load', () => {
    console.log('Page loaded');
    console.log('✅ Using simplified protocol: Raw PCM bytes + JSON');
    console.log('   No MessagePack library needed!');
    
    // Enable recording button
    recordBtn.disabled = false;
    recordBtn.textContent = 'Start Recording';
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (isRecording) {
        stopRecording();
    }
    if (ws) {
        ws.close();
    }
});

