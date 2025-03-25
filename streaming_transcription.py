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


logger = logging.getLogger(__name__)


async def connect_transcription_session():
    """Establish a WebSocket connection to the streaming transcription service
    and send the initial configuration payload.
    """
    import websockets
    from transcribe_service.config import OPENAI_API_KEY

    url = "wss://api.openai.com/v1/realtime?intent=transcription"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    ws = await websockets.connect(url, extra_headers=headers)

    # Define the initial configuration payload for the transcription session.
    # The payload includes:
    # - type: specifies the message type for the session update.
    # - input_audio_format: the format of the incoming audio (here "pcm16").
    # - input_audio_transcription: configuration for the transcription model, including model name, prompt, and language.
    # - turn_detection: VAD settings for detecting speech turns, including threshold, prefix padding, and silence duration.
    # - input_audio_noise_reduction: settings to enable noise reduction.
    # - include: list of additional data to include in the response.
    initial_payload = {
        "type": "transcription_session.update",
        "input_audio_format": "pcm16",
        "input_audio_transcription": {
            "model": "gpt-4o-transcribe",
            "prompt": "",
            "language": ""
        },
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500
        },
        "input_audio_noise_reduction": {
            "type": "near_field"
        },
        "include": ["item.input_audio_transcription.logprobs"]
    }
    await ws.send(json.dumps(initial_payload))
    logger.info("Sent initial configuration payload for transcription session.")
    return ws


async def send_audio_chunks(ws, audio_source):
    """Continuously send audio chunks from the audio_source over the WebSocket connection.

    Splits audio data into parts of max_chunk_size bytes with an overlap to ensure continuity if needed.

    Parameters:
        ws: The active WebSocket connection.
        audio_source: An asynchronous source (e.g., an asyncio.Queue) providing audio chunks.
    """
    import asyncio
    max_chunk_size = 4096
    # Set overlap size (in bytes) to provide overlapping segments if needed.
    overlap_size = 512
    while True:
        try:
            chunk = await audio_source.get()
        except Exception:
            await asyncio.sleep(0.01)
            continue
        if not chunk:
            await asyncio.sleep(0.01)
            continue
        offset = 0
        # Send chunks with an overlap between consecutive parts.
        while offset < len(chunk):
            part = chunk[offset: offset + max_chunk_size]
            try:
                await ws.send(part)
            except Exception as e:
                logger.error("Error sending audio chunk: %s", e)
                break
            offset += (max_chunk_size - overlap_size) if max_chunk_size > overlap_size else max_chunk_size


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


async def handle_incoming_transcriptions(ws):
    """Handle incoming transcription messages from the WebSocket connection.

    Parameters:
        ws: The active WebSocket connection.
    """
    transcript = ""
    try:
        async for message in ws:
            try:
                data = json.loads(message)
                partial = data.get("partial")
                final = data.get("final")
                if final:
                    transcript += final
                    logger.info("Final transcript: %s", transcript)
                elif partial:
                    logger.info("Partial transcript: %s", partial)
                else:
                    logger.info("Unknown message: %s", data)
            except Exception as e:
                logger.error("Error processing message: %s", e)
    except Exception as e:
        logger.error("Error receiving message: %s", e)


async def manage_streaming():
    """Manage the overall streaming workflow including connection,
    sending audio data, and handling incoming messages with error handling
    and reconnection logic.
    """
    ws = await connect_transcription_session()
    audio_source = asyncio.Queue()
    sending_task = asyncio.create_task(send_audio_chunks(ws, audio_source))
    receiving_task = asyncio.create_task(handle_incoming_transcriptions(ws))
    await asyncio.gather(sending_task, receiving_task)

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
