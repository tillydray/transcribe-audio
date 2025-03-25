import unittest
import asyncio
import json
from unittest.mock import AsyncMock, patch
import streaming_transcription
import logging

# A fake async websocket for iterating over messages and capturing sends.
class FakeWebSocket:
    def __init__(self, messages):
        self._messages = messages
        self.sent_messages = []

    def __aiter__(self):
        self._iter = iter(self._messages)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def send(self, message):
        self.sent_messages.append(message)

class TestStreamingTranscription(unittest.TestCase):
    def setUp(self):
        self.final_message = json.dumps({"final": "final transcript text", "partial": None})
        self.partial_message = json.dumps({"final": "", "partial": "partial transcript text"})

    def test_handle_incoming_transcriptions_final(self):
        # Test that final transcriptions trigger a log info.
        fake_ws = FakeWebSocket([self.final_message])
        # Patch logger.info to capture log calls
        with patch.object(logging.getLogger(), "info") as mock_log_info:
            async def run():
                await streaming_transcription.handle_incoming_transcriptions(fake_ws)
            # Run the coroutine; it will exit after processing the message.
            try:
                asyncio.run(run())
            except Exception:
                pass
            # Assert that logging.info was called with a transcript message
            self.assertTrue(any("Final transcript:" in call.args[0] for call in mock_log_info.call_args_list))

    def test_handle_incoming_transcriptions_partial(self):
        # Test that partial transcriptions trigger a log info.
        fake_ws = FakeWebSocket([self.partial_message])
        with patch.object(logging.getLogger(), "info") as mock_log_info:
            async def run():
                await streaming_transcription.handle_incoming_transcriptions(fake_ws)
            try:
                asyncio.run(run())
            except Exception:
                pass
            self.assertTrue(any("Partial transcript:" in call.args[0] for call in mock_log_info.call_args_list))

    @patch('streaming_transcription.websockets.connect', new_callable=AsyncMock)
    def test_connect_transcription_session(self, mock_ws_connect):
        # Create a fake websocket with an async send method.
        fake_ws = AsyncMock()
        mock_ws_connect.return_value = fake_ws

        async def run():
            ws = await streaming_transcription.connect_transcription_session()
            # Validate that an initial payload was sent on connection.
            self.assertTrue(fake_ws.send.called)
            sent_payload = json.loads(fake_ws.send.call_args[0][0])
            self.assertEqual(sent_payload.get("type"), "transcription_session.update")
            self.assertEqual(sent_payload.get("input_audio_format"), "pcm16")
        asyncio.run(run())

if __name__ == '__main__':
    unittest.main()
