import unittest
from transcribe_service.vad_processing import VoiceActivityDetector

class TestIsSpeech(unittest.TestCase):
    def test_is_speech_returns_true_when_majority_frames_speech(self):
        sample_rate = 16000
        frame_duration_ms = 30
        # Create a VoiceActivityDetector instance.
        vad_detector = VoiceActivityDetector(mode=1, frame_duration_ms=frame_duration_ms)
        # Calculate frame size.
        bytes_per_sample = 2
        num_samples_per_frame = int(sample_rate * (frame_duration_ms / 1000))
        frame_size = num_samples_per_frame * bytes_per_sample  # e.g., 960 bytes
        number_frames = 10
        total_bytes = frame_size * number_frames
        # Create a dummy audio_data consisting of total_bytes.
        audio_data = b'\x00' * total_bytes

        # Monkey-patch the vad.is_speech method to simulate speech detection.
        # First, simulate that the first 6 frames are speech, and the remaining 4 are silence.
        original_is_speech = vad_detector.vad.is_speech
        call_count = 0
        def fake_is_speech(frame: bytes, sample_rate: int) -> bool:
            nonlocal call_count
            result = True if call_count < 6 else False
            call_count += 1
            return result
        vad_detector.vad.is_speech = fake_is_speech

        # Call is_speech() on the full audio_data.
        result = vad_detector.is_speech(audio_data, sample_rate)
        # With 6 out of 10 frames marked as speech (i.e., > 50%), it should return True.
        self.assertTrue(result)

        # Now simulate that all frames are non-speech.
        call_count = 0
        def fake_is_speech_all_false(frame: bytes, sample_rate: int) -> bool:
            nonlocal call_count
            call_count += 1
            return False
        vad_detector.vad.is_speech = fake_is_speech_all_false

        result = vad_detector.is_speech(audio_data, sample_rate)
        self.assertFalse(result)

        # Restore the original method.
        vad_detector.vad.is_speech = original_is_speech

if __name__ == '__main__':
    unittest.main()
