import unittest
from transcribe_service.vad_processing import VoiceActivityDetector

class TestFrameGenerator(unittest.TestCase):
    def test_frame_generator_splits_correctly(self):
        # Set the configuration values.
        sample_rate = 16000
        frame_duration_ms = 30
        # Calculate the expected frame size:
        # Number of samples = sample_rate * (frame_duration_ms / 1000) = 480 samples
        # Each sample is 2 bytes, so frame size = 480 * 2 = 960 bytes
        expected_frame_size = 960
        # Create a dummy PCM byte string that is exactly 3 frames long.
        total_frames = 3
        total_bytes = expected_frame_size * total_frames
        # For a consistent pattern, we'll use a sequence that repeats.
        # Since bytes must be in range 0-255, we'll create a sequence and repeat it.
        pattern = bytes(range(256))
        audio_data = (pattern * ((total_bytes // 256) + 1))[:total_bytes]
        
        vad_detector = VoiceActivityDetector(mode=1, frame_duration_ms=frame_duration_ms)
        frames = list(vad_detector.frame_generator(audio_data, sample_rate))
        
        # Verify that the generator yields the expected number of frames.
        self.assertEqual(len(frames), total_frames)
        # Verify that each frame is exactly expected_frame_size bytes
        for frame in frames:
            self.assertEqual(len(frame), expected_frame_size)

if __name__ == '__main__':
    unittest.main()
