import asyncio
import json
import unittest
import logging
from transcribe_service.streaming_transcription import send_audio_chunks, handle_incoming_transcriptions

class FakeWebSocket:
    def __init__(self):
        self.sent_messages = []
        self._messages = asyncio.Queue()

    async def send(self, message):
        self.sent_messages.append(message)

    async def __aiter__(self):
        while True:
            message = await self._messages.get()
            if message is None:
                break
            yield message

    def add_incoming_message(self, message):
        self._messages.put_nowait(message)

    def close_incoming(self):
        self._messages.put_nowait(None)


class TestIntegrationStreamingTranscription(unittest.TestCase):
    def setUp(self):
        # Create a fake audio source with a prerecorded audio chunk.
        self.fake_audio_source = asyncio.Queue()
        # Use a small fake audio chunk (less than max_chunk_size) for simplicity.
        self.fake_audio_source.put_nowait(b"fake_audio_data")
        # Create a fake websocket for both sending and receiving.
        self.fake_ws = FakeWebSocket()

    def test_send_audio_chunks(self):
        async def run_test():
            # Start send_audio_chunks in a task.
            task = asyncio.create_task(send_audio_chunks(self.fake_ws, self.fake_audio_source))
            # Allow some time for the function to process the chunk.
            await asyncio.sleep(0.1)
            task.cancel()
            # Ensure at least one message was sent.
            self.assertGreater(len(self.fake_ws.sent_messages), 0)
        asyncio.run(run_test())

    def test_handle_incoming_transcriptions(self):
        async def run_test():
            # Create a transcript message to simulate a final transcription response.
            transcript_message = {
                "final": "final transcript text",
                "partial": None
            }
            # Add the encoded message to the fake websocket.
            self.fake_ws.add_incoming_message(json.dumps(transcript_message))
            # Signal the end of incoming messages.
            self.fake_ws.close_incoming()
            # Setup a stream handler to capture log output.
            import io
            log_stream = io.StringIO()
            handler = logging.StreamHandler(log_stream)
            logger = logging.getLogger(__name__)
            logger.addHandler(handler)
            await handle_incoming_transcriptions(self.fake_ws)
            handler.flush()
            logs = log_stream.getvalue()
            self.assertIn("Final transcript: final transcript text", logs)
            logger.removeHandler(handler)
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
