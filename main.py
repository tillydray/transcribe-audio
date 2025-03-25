#/usr/bin/env python3
"""
Module for capturing audio from an input stream, converting it to WAV format,
and transcribing it via the OpenAI API.
"""

import sounddevice as sd
import io
import os
import queue
import threading
import time
import wave
import numpy as np
from openai import OpenAI
import vad
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
audio_queue = queue.Queue()

DEVICE_NAME = "Aggregate Device"
CHANNELS = 1
SAMPLERATE = 16000
SEGMENT_SECONDS = 5  # Collect 5 seconds of audio for each transcription
vad_detector = vad.VoiceActivityDetector(mode=1, frame_duration_ms=30)


def enque_audio(indata, frames, time_info, status):
    """Enqueue a copy of the incoming audio data into the global audio_queue."""
    if status:
        print("Streaming status:", status)
    # Simply enqueue the raw audio data (as a copy to avoid conflicts)
    audio_queue.put(indata.copy())


def process_audio_segment(topic):
    """Accumulate audio data and call the transcription API.
    
    Arguments:
        topic (str): The transcription topic to include in the prompt.
    """
    # Calculate the number of frames we need for SEGMENT_SECONDS of audio.
    frames_per_segment = SAMPLERATE * SEGMENT_SECONDS
    prev_transcript = ""
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
            # Convert each chunk from float32 to int16 and concatenate
            processed_chunks = [
                (chunk * 32767).astype(np.int16).tobytes() for chunk in segments
            ]
            audio_data = b"".join(processed_chunks)
            # Generate frames using the VAD's frame_generator.
            frames = list(vad_detector.frame_generator(audio_data, SAMPLERATE))
            if not frames:
                print("No frames generated, skipping segment.")
                continue
            # Use vad_collector to yield only voiced segments, with 300 ms padding.
            padded_voiced_segments = list(
                vad.vad_collector(SAMPLERATE, vad_detector.frame_duration_ms, 300, vad_detector.vad, frames)
            )
            if not padded_voiced_segments:
                print("Silence detected (via vad_collector), skipping transcription for this segment.")
                continue
            # Concatenate the voiced segments.
            voiced_audio = b"".join(padded_voiced_segments)
            # Write the voiced audio into an in-memory WAV file with a proper header.
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(SAMPLERATE)
                wf.writeframes(voiced_audio)
            wav_buffer.seek(0)
            # Prepare file tuple. Some APIs expect (filename, fileobj, mimetype)
            file_tuple = ("audio.wav", wav_buffer, "audio/wav")
            # Build the prompt dynamically including the transcription topic.
            if prev_transcript:
                prompt = (
                    f"Topic: {topic}\n"
                    f"Previous transcript: {prev_transcript}\n"
                    "Now, transcribe the current audio segment with proper punctuation and clarity:"
                )
            else:
                prompt = (
                    f"Topic: {topic}\n"
                    "Transcribe the current audio segment with proper punctuation and clarity:"
                )
            try:
                response = client.audio.transcriptions.create(
                    file=file_tuple,
                    model="gpt-4o-transcribe",
                    language="en",
                    prompt=prompt,
                    temperature=0
                )
                current_transcript = response.text
                print("Transcription:", current_transcript)
                prev_transcript = current_transcript
            except Exception as e:
                print("Error calling transcription API:", e)
        else:
            # If no segments accumulated, sleep briefly
            time.sleep(0.1)


def main():
    """Start the audio processing worker thread and initiates the audio stream."""
    # Prompt user for transcription topic
    topic = input("Enter the transcription topic (press Enter for a generic topic): ")
    if not topic.strip():
        topic = "general conversation"
        print("Using default topic: 'general conversation'")
    else:
        print(f"Using topic: '{topic}'")
    worker = threading.Thread(target=process_audio_segment, args=(topic,), daemon=True)
    worker.start()

    # Prompt user for audio device selection
    devices = sd.query_devices()
    all_input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    default_input_idx = sd.default.device[0]
    reindexed_devices = [(new_idx, orig_idx, dev) for new_idx, (orig_idx, dev) in enumerate(all_input_devices)]
    print("Available input devices:")
    for new_idx, orig_idx, dev in reindexed_devices:
        default_str = " (default)" if orig_idx == default_input_idx else ""
        print(f"  {new_idx}: {dev['name']}{default_str}")
    choice = input("Select an audio device (press Enter for default): ")
    if choice.strip() == "":
        device_to_use_index = default_input_idx
        print("Using default input device.")
    else:
        try:
            new_idx = int(choice)
            orig_idx = reindexed_devices[new_idx][1]
            device_to_use_index = orig_idx
        except Exception as e:
            print("Invalid selection, using default input device.")
            device_to_use_index = default_input_idx
    device_to_use = sd.query_devices(device_to_use_index)['name']

    try:
        with sd.InputStream(
            device=device_to_use,
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
