import openai
import internal_logging as logging
import os

logger = logging.logger
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def transcribe_audio(file_tuple, prompt: str, language: str) -> str:
    import time
    max_attempts = 3
    delay = 1
    attempt = 0
    while attempt < max_attempts:
        try:
            response = client.audio.transcriptions.create(
                file=file_tuple,
                model="gpt-4o-transcribe",
                language=language,
                prompt=prompt,
                temperature=0
            )
            return response.text
        except Exception as e:
            logger.error("Error calling transcription API (attempt %d): %s", attempt + 1, e)
            attempt += 1
            time.sleep(delay)
            delay *= 2
    logger.error("Transcription API failed after %d attempts", max_attempts)
    return ""

def generate_topic_from_context(full_transcript: str, initial_topic: str, previous_topic: str, language: str) -> str:
    import time
    prompt = (
        f"Based on the following conversation transcript:\n{full_transcript}\n"
        f"The user initially indicated the topic as '{initial_topic}', and the previous refined topic was '{previous_topic}'.\n"
        "Please generate a refined, concise topic that best represents the ongoing conversation:"
    )
    max_attempts = 3
    delay = 1
    attempt = 0
    while attempt < max_attempts:
        try:
            response = client.completions.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=30,
                temperature=0.5,
                language=language
            )
            new_topic = response.choices[0].text.strip()
            return new_topic
        except Exception as e:
            logger.error("Error generating topic from context (attempt %d): %s", attempt + 1, e)
            attempt += 1
            time.sleep(delay)
            delay *= 2
    logger.error("Topic generation failed after %d attempts", max_attempts)
    return previous_topic
