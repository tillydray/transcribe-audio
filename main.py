# /usr/bin/env python3
"""
Module for capturing audio from an input stream, converting it to WAV format,
and transcribing it via the OpenAI API.
"""

import sounddevice as sd
import threading
import queue
import time
import io
import wave
import numpy as np
import internal_logging as logging
from transcribe_service.audio_capture import start_audio_capture, list_input_devices, audio_queue
from transcribe_service.vad_processing import VoiceActivityDetector, vad_collector
from transcribe_service.api_client import transcribe_audio, generate_topic_from_context
from transcribe_service.config import LANGUAGE_CODE, CHANNELS, SAMPLERATE, SEGMENT_SECONDS
from streaming_transcription import manage_streaming_with_reconnect

logger = logging.logger
vad_detector = VoiceActivityDetector(mode=1, frame_duration_ms=30)


def process_audio_segment(initial_topic: str) -> None:
    """Process audio segments by collecting audio frames from the queue, applying VAD, and transcribing speech.

    Parameters:
        initial_topic (str): The initial topic for transcription.
    """
    current_topic = initial_topic
    full_transcript = ""
    last_topic_update = time.time()
    refinements = 0
    frames_per_segment = SAMPLERATE * SEGMENT_SECONDS
    prev_transcript = ""
    while True:
        collected_frames = 0
        segments = []
        while collected_frames < frames_per_segment:
            try:
                data = audio_queue.get(timeout=1)
                segments.append(data)
                collected_frames += data.shape[0]
            except queue.Empty:
                if segments:
                    break
        if segments:
            processed_chunks = [
                (chunk * 32767).astype(np.int16).tobytes() for chunk in segments
            ]
            audio_data = b"".join(processed_chunks)
            frames = list(vad_detector.frame_generator(audio_data, SAMPLERATE))
            if not frames:
                logger.info("No frames generated, skipping segment.")
                continue
            padded_voiced_segments = list(
                vad_collector(
                    SAMPLERATE,
                    vad_detector.frame_duration_ms,
                    300,
                    vad_detector.vad,
                    frames)
            )
            if not padded_voiced_segments:
                logger.info(
                    "Silence detected (via vad_collector), skipping transcription for this segment.")
                continue
            voiced_audio = b"".join(padded_voiced_segments)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLERATE)
                wf.writeframes(voiced_audio)
            wav_buffer.seek(0)
            file_tuple = ("audio.wav", wav_buffer, "audio/wav")
            if prev_transcript:
                prompt = (
                    f"Topic: {current_topic}\n"
                    f"Previous transcript: {prev_transcript}\n"
                    "Now, transcribe the current audio segment with proper punctuation and clarity:"
                )
            else:
                prompt = (
                    f"Topic: {current_topic}\n"
                    "Transcribe the current audio segment with proper punctuation and clarity:"
                )
            current_transcript = transcribe_audio(file_tuple, prompt, LANGUAGE_CODE)
            if current_transcript:
                print("Transcription:", current_transcript)
                prev_transcript = current_transcript
                full_transcript += "\n" + current_transcript
                now = time.time()
                if now - last_topic_update >= 60 and refinements < 10:
                    new_topic = generate_topic_from_context(
                        full_transcript, initial_topic, current_topic, LANGUAGE_CODE)
                    logger.info("Refined topic: %s", new_topic)
                    current_topic = new_topic
                    last_topic_update = now
                    refinements += 1
        else:
            time.sleep(0.1)


def main() -> None:
    """Main entry point for the transcription application.

    Prompts for the transcription topic, starts the audio processing thread, lists available audio devices,
    and begins audio capture.
    """
    mode = input("Select transcription mode - Batch (B) or Streaming (S): ")
    if mode.lower() == "s":
        print("Starting streaming transcription mode.")
        import asyncio
        asyncio.run(manage_streaming_with_reconnect())
        return
    topic = input("Enter the transcription topic (press Enter for a generic topic): ")
    if not topic.strip():
        topic = "general conversation"
        logger.info("Using default topic: 'general conversation'")
    else:
        logger.info("Using topic: '%s'", topic)
    worker = threading.Thread(target=process_audio_segment, args=(topic,), daemon=True)
    worker.start()

    devices = list_input_devices()
    default_input_idx = sd.default.device[0]
    reindexed_devices = [(new_idx, orig_idx, dev)
                         for new_idx, (orig_idx, dev) in enumerate(devices)]
    print("Available input devices:")
    for new_idx, orig_idx, dev in reindexed_devices:
        default_str = " (default)" if orig_idx == default_input_idx else ""
        print(f"  {new_idx}: {dev['name']} {default_str}")
    choice = input("Select an audio device (press Enter for default): ")
    if choice.strip() == "":
        device_to_use_index = default_input_idx
        print("Using default input device.")
    else:
        try:
            new_idx = int(choice)
            orig_idx = reindexed_devices[new_idx][1]
            device_to_use_index = orig_idx
        except Exception:
            logger.warning("Invalid selection, using default input device.")
            device_to_use_index = default_input_idx
    device_to_use = sd.query_devices(device_to_use_index)['name']
    start_audio_capture(device_to_use, CHANNELS, SAMPLERATE)


if __name__ == '__main__':
    main()
