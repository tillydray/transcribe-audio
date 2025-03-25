"""Module for configuration constants for the audio transcription service."""
import dotenv
import locale
import os

dotenv.load_dotenv()

cur_locale = locale.getlocale()
LANGUAGE_CODE = "en" if cur_locale[0] is None else cur_locale[0].split("_")[0].lower()

DEVICE_NAME = "Aggregate Device"
CHANNELS = 1
SAMPLERATE = 16000
SEGMENT_SECONDS = 5  # Collect 5 seconds of audio for each transcription
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
STREAMING_MODEL = "gpt-4o-transcribe"
INPUT_AUDIO_FORMAT = "pcm16"
STREAMING_PROMPT = ""
STREAMING_THRESHOLD = 0.5
STREAMING_PREFIX_PADDING_MS = 300
STREAMING_SILENCE_DURATION_MS = 500
