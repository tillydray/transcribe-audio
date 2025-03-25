import webrtcvad

class VoiceActivityDetector:
    def __init__(self, mode=2, frame_duration_ms=30):
        """
        Initialize the VAD.
        
        Parameters:
            mode (int): Aggressiveness mode (0-3).
            frame_duration_ms (int): Duration of each frame in ms.
        """
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(mode)
        self.frame_duration_ms = frame_duration_ms

    def frame_generator(self, audio, sample_rate):
        """
        Generator that yields audio frames of fixed duration.
        
        Parameters:
            audio (bytes): Raw PCM audio data (16-bit little-endian).
            sample_rate (int): Sample rate of the audio.
        
        Yields:
            bytes: A frame of audio data.
        """
        bytes_per_sample = 2
        num_samples_per_frame = int(sample_rate * (self.frame_duration_ms / 1000.0))
        frame_size = num_samples_per_frame * bytes_per_sample
        for offset in range(0, len(audio), frame_size):
            if offset + frame_size > len(audio):
                break
            yield audio[offset:offset+frame_size]

    def is_speech(self, audio, sample_rate):
        """
        Determine whether the given audio contains speech.
        
        Parameters:
            audio (bytes): Raw PCM 16-bit audio data.
            sample_rate (int): Sample rate of the audio.
        
        Returns:
            bool: True if more than half of the audio frames are speech.
        """
        frames = list(self.frame_generator(audio, sample_rate))
        if not frames:
            return False
        
        speech_frames = 0
        for frame in frames:
            if self.vad.is_speech(frame, sample_rate):
                speech_frames += 1
                
        return speech_frames > len(frames) / 2
