#!/usr/bin/env python3
import unittest
import numpy as np
from transcribe_service.audio_capture import enque_audio, audio_queue

class TestEnqueAudio(unittest.TestCase):
    def test_enque_audio_places_copy_in_queue(self) -> None:
        # Clear the queue before testing
        while not audio_queue.empty():
            audio_queue.get()

        # Create a small NumPy array
        test_array = np.array([1, 2, 3, 4], dtype=np.float32)
        # Call enque_audio function with dummy values for frames, time_info, and status
        enque_audio(test_array, 4, {}, None)

        # Verify the queue now contains one item
        self.assertFalse(audio_queue.empty(), "audio_queue should have one element")
        queued_item = audio_queue.get()

        # Check that the queued item is equal to the original array data
        np.testing.assert_array_equal(queued_item, test_array)

        # Modify the original array to ensure the enqueued item is a copy
        test_array[0] = 99
        # The enqueued item should remain unchanged
        self.assertNotEqual(queued_item[0], test_array[0], "Enqueued audio should be a copy, not a reference")

if __name__ == '__main__':
    unittest.main()
