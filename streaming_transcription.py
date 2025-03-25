"""Module for streaming transcription using WebSocket connection.

This module implements functions to:
- Establish a connection to the streaming transcription service.
- Send audio chunks continuously.
- Handle incoming transcription messages.
- Manage error handling and reconnections.

Each function below is a placeholder to be implemented in subsequent steps.
"""

import asyncio
import json
import logging
import websockets


logger = logging.getLogger(__name__)


async def connect_transcription_session():
    """Establish a WebSocket connection to the streaming transcription service
    and send the initial configuration payload.
    """
    from transcribe_service.config import OPENAI_API_KEY, INPUT_AUDIO_FORMAT, STREAMING_MODEL, STREAMING_PROMPT, STREAMING_THRESHOLD, STREAMING_PREFIX_PADDING_MS, STREAMING_SILENCE_DURATION_MS, LANGUAGE_CODE

    url = "wss://api.openai.com/v1/realtime?intent=transcription"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "openai-beta": "realtime=v1"
    }
    ws = await websockets.connect(url, additional_headers=headers)

    # Define the initial configuration payload for the transcription session.
    # The payload includes:
    # - type: specifies the message type for the session update.
    # - input_audio_format: uses the configured audio format.
    # - input_audio_transcription: configuration for the transcription model using configured model, prompt, and language.
    # - turn_detection: VAD settings for detecting speech turns using configured parameters.
    # - input_audio_noise_reduction: settings to enable noise reduction.
    # - include: list of additional data to include in the response.
    initial_payload = {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": INPUT_AUDIO_FORMAT,
            "input_audio_transcription": {
                "model": STREAMING_MODEL,
                "prompt": STREAMING_PROMPT,
                "language": LANGUAGE_CODE
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": STREAMING_THRESHOLD,
                "prefix_padding_ms": STREAMING_PREFIX_PADDING_MS,
                "silence_duration_ms": STREAMING_SILENCE_DURATION_MS
            },
            "input_audio_noise_reduction": {
                "type": "near_field"
            },
            "include": ["item.input_audio_transcription.logprobs"]
        }}
    await ws.send(json.dumps(initial_payload))
    logger.info("Sent initial configuration payload for transcription session.")
    return ws


async def send_audio_chunks(ws, audio_source):
    """Continuously send audio chunks from the audio_source over the WebSocket connection.

    Each audio chunk is base64 encoded and sent as a JSON message with type 'input_audio_buffer.append'.
    If a session ID is available, it is included in the payload.

    Parameters:
        ws: The active WebSocket connection.
        audio_source: An asynchronous source (e.g., an asyncio.Queue) providing audio chunks.
    """
    import asyncio
    import base64
    import json
    max_chunk_size = 4096
    overlap_size = 512
    while True:
        try:
            chunk = await audio_source.get()
        except Exception:
            await asyncio.sleep(0.01)
            continue
        if chunk is None or (hasattr(chunk, "size") and chunk.size == 0):
            await asyncio.sleep(0.01)
            continue
        offset = 0
        while offset < len(chunk):
            part = chunk[offset: offset + max_chunk_size]
            # Encode the audio data in base64
            audio_b64 = base64.b64encode(part).decode('utf-8')
            payload = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
            try:
                await ws.send(json.dumps(payload))
            except Exception as e:
                logger.error("Error sending audio chunk: %s", e)
                break
            offset += (max_chunk_size -
                       overlap_size) if max_chunk_size > overlap_size else max_chunk_size


async def audio_buffer_generator(audio_source):
    """Generate audio buffers from an audio source.

    This async generator retrieves audio chunks from an asynchronous source (e.g., asyncio.Queue)
    and converts them to PCM byte format suitable for streaming.

    Parameters:
        audio_source: An asynchronous source (e.g., asyncio.Queue) that yields audio chunks as NumPy arrays.

    Yields:
        bytes: PCM audio data in bytes (16-bit little-endian).
    """
    import numpy as np
    while True:
        chunk = await audio_source.get()
        pcm_bytes = (chunk * 32767).astype(np.int16).tobytes()
        yield pcm_bytes


async def audio_bridge(audio_source):
    """Bridge the global audio_queue from audio_capture into the async audio_source queue."""
    from transcribe_service.audio_capture import audio_queue
    import asyncio
    while True:
        if not audio_queue.empty():
            item = audio_queue.get()
            await audio_source.put(item)
        else:
            await asyncio.sleep(0.01)


async def handle_incoming_transcriptions(ws):
    """Handle incoming transcription messages from the WebSocket connection.

    Parameters:
        ws: The active WebSocket connection.
    """
    import json
    transcript = ""
    try:
        async for message in ws:
            try:
                data = json.loads(message)
                event_type = data.get("type")
                if event_type == "transcription_session.created":
                    logger.info("Session created: %s", data)
                    continue
                elif event_type == "transcription_session.updated":
                    logger.info("Session updated: %s", data)
                    continue
                elif event_type == "conversation.item.input_audio_transcription.delta":
                    delta = data.get("delta", "")
                    transcript += delta
                    print("Delta transcription:", delta)
                    continue
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    final_transcript = data.get("transcript", "")
                    transcript += final_transcript
                    print("Final transcription:", transcript)
                    transcript = ""
                    continue
                else:
                    logger.info("Unknown message: %s", data)
            except Exception as e:
                print("Error processing message:", e)
    except Exception as e:
        print("Error receiving messages:", e)


async def manage_streaming():
    """Manage the overall streaming workflow including connection,
    sending audio data, and handling incoming messages with error handling
    and reconnection logic.
    """
    ws = await connect_transcription_session()
    audio_source = asyncio.Queue()
    bridge_task = asyncio.create_task(audio_bridge(audio_source))
    sending_task = asyncio.create_task(send_audio_chunks(ws, audio_source))
    receiving_task = asyncio.create_task(handle_incoming_transcriptions(ws))
    await asyncio.gather(sending_task, receiving_task, bridge_task)


async def manage_streaming_with_reconnect():
    """Manage the streaming workflow with automatic reconnection using exponential backoff.

    This function wraps the connection and streaming logic in a resilient loop. If the session
    fails due to an exception, it will wait for an exponentially increasing delay before reconnecting.
    """
    delay = 1
    max_delay = 60
    while True:
        try:
            logging.getLogger(__name__).info("Starting streaming session.")
            await manage_streaming()
        except Exception as e:
            logging.getLogger(__name__).error("Streaming session terminated unexpectedly: %s", e)
        logging.getLogger(__name__).info("Reconnecting in %d seconds...", delay)
        await asyncio.sleep(delay)
        delay = min(delay * 2, max_delay)
