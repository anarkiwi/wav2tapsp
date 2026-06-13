# Copyright 2022 Josh Bailey (josh@vandervecken.com)

"""wav2tapsp: convert WAV recordings of C64 cassette tapes into TAP files."""

from wav2tapsp.convert import (
    DEFAULT_CPUFREQ,
    pulses_to_tap,
    read_wav_to_tap,
    wav_to_pulses,
    wav_to_tap,
)

__version__ = '0.2.0'

__all__ = [
    'DEFAULT_CPUFREQ',
    'pulses_to_tap',
    'read_wav_to_tap',
    'wav_to_pulses',
    'wav_to_tap',
    '__version__',
]
