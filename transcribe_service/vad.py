"""Module for Voice Activity Detection (VAD) and audio frame processing."""
import webrtcvad
from typing import Generator, Iterable
import collections
import internal_logging as logging

logger = logging.logger


class VoiceActivityDetector:
    def __init__(self, mode: int = 1, frame_duration_ms: int = 30) -> None:
        """
        Initialize the VAD.

        Parameters:
            mode (int): Aggressiveness mode (0-3).
            frame_duration_ms (int): Duration of each frame in ms.
        """
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(mode)
        self.frame_duration_ms = frame_duration_ms

    def frame_generator(self, audio: bytes, sample_rate: int) -> Generator[bytes, None, None]:
        """
        Generate audio frames of fixed duration.

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
            yield audio[offset:offset + frame_size]

    def is_speech(self, audio: bytes, sample_rate: int) -> bool:
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


def vad_collector(sample_rate: int, frame_duration_ms: int,
                  padding_duration_ms: int, vad: webrtcvad.Vad, frames: Iterable[bytes]) -> Generator[bytes, None, None]:
    """
    Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator) where each frame is raw PCM bytes.

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        logger.debug('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                logger.debug('+')
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                logger.debug('-')
                triggered = False
                yield b''.join(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        logger.debug('-')
    logger.debug('\n')
    if voiced_frames:
        yield b''.join(voiced_frames)
