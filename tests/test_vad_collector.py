import unittest
from transcribe_service.vad_processing import vad_collector
import webrtcvad

class FakeVAD:
    """
    A fake VAD to simulate is_speech responses based on a predefined pattern.
    The speech_pattern is a list of booleans in the order of the frames.
    """
    def __init__(self, speech_pattern):
        self.speech_pattern = speech_pattern
        self.index = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        if self.index < len(self.speech_pattern):
            result = self.speech_pattern[self.index]
            self.index += 1
            return result
        return False

class TestVadCollector(unittest.TestCase):
    def test_vad_collector_yields_voiced_segment(self):
        # Simulate six frames with the following speech pattern:
        # Frame 0: False, Frame 1: True, Frame 2: True, Frame 3: True, Frame 4: False, Frame 5: False.
        # We expect the vad_collector to trigger on frames 1-3.
        fake_frames = [b'frame_%d' % i for i in range(6)]
        speech_pattern = [False, True, True, True, False, False]
        fake_vad = FakeVAD(speech_pattern)
        
        sample_rate = 16000
        frame_duration_ms = 30
        padding_duration_ms = 90  # This equals 3 frames
        
        # Reset fake_vad index.
        fake_vad.index = 0
        
        segments = list(vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, fake_vad, fake_frames))
        
        # We expect at least one segment to be returned
        self.assertTrue(len(segments) >= 1)
        
        # Expected voiced segment: concatenation of fake_frames[1], fake_frames[2], fake_frames[3]
        expected_voiced = b''.join(fake_frames[1:4])
        
        # Check that the expected voiced bytes appear in the first segment.
        self.assertIn(expected_voiced, segments[0])

if __name__ == '__main__':
    unittest.main()
