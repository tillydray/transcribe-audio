import sounddevice as sd
import numpy as np
import time
import queue
import internal_logging as logging

logger = logging.logger
audio_queue = queue.Queue()

def enque_audio(indata: np.ndarray, frames: int, time_info: dict, status: object) -> None:
    """Enqueue a copy of the incoming audio data into the global audio_queue."""
    if status:
        logger.debug("Streaming status: %s", status)
    # Simply enqueue the raw audio data (as a copy to avoid conflicts)
    audio_queue.put(indata.copy())

def list_input_devices():
    devices = sd.query_devices()
    all_input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    return all_input_devices

def start_audio_capture(device_name: str, channels: int, samplerate: int):
    try:
        with sd.InputStream(
            device=device_name,
            channels=channels,
            samplerate=samplerate,
            callback=enque_audio
        ):
            print("Collecting audio for transcription. Press Ctrl+C to exit...")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        logger.error("Stream error: %s", e)
