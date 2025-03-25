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
