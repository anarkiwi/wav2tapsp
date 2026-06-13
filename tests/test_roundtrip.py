"""Full round-trip tests that exercise wav2tapsp on synthesised audio.

    PRG  ->  TAP  ->  WAV  --(wav2tapsp)-->  TAP  ->  PRG

The recovered PRG must be byte-identical to the original and must still be a
structurally valid, runnable C64 BASIC program.
"""

import os
import subprocess
import sys

import numpy as np

import c64tape
import wav2tapsp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 96 kHz gives a comfortable timing margin for the short/medium/long pulse
# classification after the lossy WAV -> TAP re-quantisation.
SAMPLERATE = 96000


def _make_wav_for(prg, path):
    pulses = c64tape.parse_tap(c64tape.prg_to_tap(prg))
    samples = c64tape.tap_pulses_to_samples(pulses, SAMPLERATE)
    c64tape.write_wav(path, samples, SAMPLERATE)


def test_roundtrip_in_process():
    prg = c64tape.sample_basic_prg()
    expected_lines = c64tape.parse_basic_prg(prg)

    # PRG -> TAP -> WAV samples
    pulses = c64tape.parse_tap(c64tape.prg_to_tap(prg))
    samples = c64tape.tap_pulses_to_samples(pulses, SAMPLERATE)

    # WAV -> TAP via the code under test
    tap = wav2tapsp.wav_to_tap(samples, SAMPLERATE)

    # TAP -> PRG (the "final C64 program")
    recovered = c64tape.tap_to_prg(tap)

    assert recovered == prg
    assert c64tape.parse_basic_prg(recovered) == expected_lines


def test_roundtrip_via_cli(tmp_path):
    prg = c64tape.sample_basic_prg()
    wav_path = str(tmp_path / 'roundtrip.wav')
    _make_wav_for(prg, wav_path)

    # Run the actual command-line entry point, exactly as a user would.
    subprocess.run(
        [sys.executable, os.path.join(ROOT, 'wav2tapsp.py'), wav_path],
        check=True, cwd=ROOT)

    tap_path = wav_path[:-4] + '.tap'
    assert os.path.exists(tap_path)
    with open(tap_path, 'rb') as f:
        tap = f.read()

    recovered = c64tape.tap_to_prg(tap)
    assert recovered == prg
    assert c64tape.parse_basic_prg(recovered) == c64tape.parse_basic_prg(prg)


def test_roundtrip_arbitrary_payload():
    # A non-BASIC payload (machine-code-like bytes) round-trips through audio too.
    import struct
    prg = struct.pack('<H', 0xC000) + bytes((i * 37 + 11) & 0xff for i in range(160))
    pulses = c64tape.parse_tap(c64tape.prg_to_tap(prg, file_type=3))
    samples = c64tape.tap_pulses_to_samples(pulses, SAMPLERATE)
    tap = wav2tapsp.wav_to_tap(samples, SAMPLERATE)
    assert c64tape.tap_to_prg(tap) == prg


def test_constant_pulse_recovered():
    # wav2tapsp should recover a constant-pulse tone to (about) that pulse length.
    for nominal in (c64tape.SHORT, c64tape.MEDIUM, c64tape.LONG):
        samples = c64tape.tap_pulses_to_samples([nominal] * 300, SAMPLERATE)
        recovered = wav2tapsp.wav_to_pulses(samples, SAMPLERATE)
        steady = recovered[5:-5]
        assert abs(float(np.median(steady)) - nominal) <= 1
