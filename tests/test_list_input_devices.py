import unittest
import sounddevice as sd
from transcribe_service.audio_capture import list_input_devices

class TestListInputDevices(unittest.TestCase):
    def test_only_input_devices_returned(self) -> None:
        # Create a fake response for sd.query_devices()
        fake_devices = [
            {'name': 'Device 1', 'max_input_channels': 2},
            {'name': 'Device 2', 'max_input_channels': 0},
            {'name': 'Device 3', 'max_input_channels': 1},
            {'name': 'Device 4', 'max_input_channels': 0},
            {'name': 'Device 5', 'max_input_channels': 4},
        ]
        # Monkey-patch sd.query_devices to return fake_devices.
        original_query_devices = sd.query_devices
        sd.query_devices = lambda: fake_devices
        try:
            result = list_input_devices()
            # Expect only devices with input channels (max_input_channels > 0) to be included.
            expected = [
                (0, {'name': 'Device 1', 'max_input_channels': 2}),
                (2, {'name': 'Device 3', 'max_input_channels': 1}),
                (4, {'name': 'Device 5', 'max_input_channels': 4}),
            ]
            self.assertEqual(result, expected)
        finally:
            # Restore the original sd.query_devices function.
            sd.query_devices = original_query_devices

if __name__ == '__main__':
    unittest.main()
