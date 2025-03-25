#!/usr/bin/env python3
"""
Module for capturing audio from an input stream, converting it to WAV format, and transcribing it via the OpenAI API.
"""

import sounddevice as sd
import io
import os
import queue
import threading
import time
import wave
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
audio_queue = queue.Queue()

DEVICE_NAME = "Aggregate Device"
CHANNELS = 1
SAMPLERATE = 16000
SEGMENT_SECONDS = 5  # Collect 5 seconds of audio for each transcription


def enque_audio(indata, frames, time_info, status):
    """Enqueue a copy of the incoming audio data into the global audio_queue."""
    if status:
        print("Streaming status:", status)
    # Simply enqueue the raw audio data (as a copy to avoid conflicts)
    audio_queue.put(indata.copy())


def process_audio_segment():
    """Accumulate audio data and call the transcription API."""
    # Calculate the number of frames we need for SEGMENT_SECONDS of audio.
    frames_per_segment = SAMPLERATE * SEGMENT_SECONDS
    while True:
        collected_frames = 0
        segments = []
        # Collect enough chunks to sum up to desired segment length.
        while collected_frames < frames_per_segment:
            try:
                data = audio_queue.get(timeout=1)
                segments.append(data)
                collected_frames += data.shape[0]
            except queue.Empty:
                # If no new data arrives, check if we have any accumulated data
                if segments:
                    break
        if segments:
            # Concatenate all collected frames
            audio_data = b"".join(chunk.tobytes() for chunk in segments)
            # Write into an in-memory WAV file with a proper header
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # assuming 16-bit PCM
                wf.setframerate(SAMPLERATE)
                wf.writeframes(audio_data)
            wav_buffer.seek(0)
            # Prepare file tuple. Some APIs expect (filename, fileobj, mimetype)
            file_tuple = ("audio.wav", wav_buffer, "audio/wav")
            try:
                response = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=file_tuple
                )
                print("Transcription:", response.text)
            except Exception as e:
                print("Error calling transcription API:", e)
        else:
            # If no segments accumulated, sleep briefly
            time.sleep(0.1)


def main():
    """Start the audio processing worker thread and initiates the audio stream."""
    worker = threading.Thread(target=process_audio_segment, daemon=True)
    worker.start()

    try:
        with sd.InputStream(
            device=DEVICE_NAME,
            channels=CHANNELS,
            samplerate=SAMPLERATE,
            callback=enque_audio
        ):
            print("Collecting audio for transcription. Press Ctrl+C to exit...")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print("Stream error:", e)


if __name__ == '__main__':
    main()
