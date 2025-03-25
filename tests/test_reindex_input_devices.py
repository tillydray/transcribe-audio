import unittest
import sounddevice as sd
from transcribe_service.audio_capture import list_input_devices

class TestReindexInputDevices(unittest.TestCase):
    def test_reindexing_and_default_marking(self) -> None:
        # Fake device list simulating sd.query_devices() output
        fake_devices = [
            {'name': 'Device A', 'max_input_channels': 2},
            {'name': 'Device B', 'max_input_channels': 0},
            {'name': 'Device C', 'max_input_channels': 1},
            {'name': 'Device D', 'max_input_channels': 0},
            {'name': 'Device E', 'max_input_channels': 4},
        ]
        # Expected filtered devices: only those with input channels (indices 0, 2, 4)
        expected_filtered = [
            (0, {'name': 'Device A', 'max_input_channels': 2}),
            (2, {'name': 'Device C', 'max_input_channels': 1}),
            (4, {'name': 'Device E', 'max_input_channels': 4}),
        ]
        # Monkey-patch sd.query_devices to return fake_devices.
        original_query_devices = sd.query_devices
        sd.query_devices = lambda: fake_devices
        
        try:
            devices = list_input_devices()
            self.assertEqual(devices, expected_filtered)
            # Reindex devices as in main.py: new indices 0, 1, 2 with original indices preserved.
            reindexed_devices = [(new_idx, orig_idx, dev) for new_idx, (orig_idx, dev) in enumerate(devices)]
            expected_reindexed = [
                (0, 0, {'name': 'Device A', 'max_input_channels': 2}),
                (1, 2, {'name': 'Device C', 'max_input_channels': 1}),
                (2, 4, {'name': 'Device E', 'max_input_channels': 4}),
            ]
            self.assertEqual(reindexed_devices, expected_reindexed)
            
            # Simulate default input device index.
            # For example, if sd.default.device[0] is 2 then the device with original index 2 is default.
            sd.default.device = (2, 1)  # default input index is 2.
            default_input_idx = sd.default.device[0]
            # Mark default if original index equals default_input_idx.
            markers = [" (default)" if orig_idx == default_input_idx else "" 
                       for new_idx, orig_idx, dev in reindexed_devices]
            expected_markers = ["", " (default)", ""]
            self.assertEqual(markers, expected_markers)
        finally:
            sd.query_devices = original_query_devices

if __name__ == '__main__':
    unittest.main()
