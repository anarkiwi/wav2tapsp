"""Round-trip coverage for the C64 fast / turbo loader formats.

Shows wav2tapsp passes each loader's pulse-timing scheme through cleanly:

    PRG  ->  turbo TAP  ->  WAV  --(wav2tapsp)-->  TAP  ->  PRG
"""

import struct

import pytest

import wav2tapsp
from wav2tapsp import analysis, c64tape, formats

# Comfortably above the fastest loader's reliable sample rate (~82 kHz), so this
# suite only asks "does the scheme survive at all"; test_sample_rate.py probes
# the actual minimums.
SR = 192000

FORMAT_IDS = [f.name for f in formats.FORMATS]
LOADER_IDS = [loader.name for loader in formats.ALL_LOADERS]


def _prg():
    return c64tape.sample_basic_prg()


@pytest.mark.parametrize('fmt', formats.FORMATS, ids=FORMAT_IDS)
def test_turbo_encode_decode_without_audio(fmt):
    prg = _prg()
    tap = formats.turbo_prg_to_tap(prg, fmt)
    # the .tap really only uses this loader's two pulse widths (plus dropped 0s)
    assert set(c64tape.parse_tap(tap)) <= {fmt.short_tap, fmt.long_tap}
    assert formats.turbo_tap_to_prg(tap, fmt) == prg


@pytest.mark.parametrize('loader', formats.ALL_LOADERS, ids=LOADER_IDS)
def test_loader_roundtrips_through_wav(loader):
    prg = _prg()
    recovered = analysis.roundtrip(loader, prg, SR)
    assert recovered == prg


def test_turbo_wrong_format_does_not_decode():
    # Decoding with mismatched pulse thresholds should not silently "succeed".
    prg = _prg()
    fast = formats.get('turrican')
    slow = formats.get('firebird')
    tap = formats.turbo_prg_to_tap(prg, fast)
    assert formats.turbo_tap_to_prg(tap, fast) == prg
    with pytest.raises(c64tape.DecodeError):
        formats.turbo_tap_to_prg(tap, slow)


def test_registry_spans_a_speed_range():
    gaps = {f.name: f.gap_cycles for f in formats.FORMATS}
    # turrican has the smallest pulse-length gap -> hardest to digitise
    assert gaps['turrican'] == min(gaps.values())
    # firebird has the widest pulses -> easiest
    assert gaps['firebird'] == max(gaps.values())


def test_arbitrary_payload_through_fastloader():
    prg = struct.pack('<H', 0xC000) + bytes((i * 37 + 11) & 0xff for i in range(120))
    fmt = formats.get('freeload')
    samples = c64tape.tap_pulses_to_samples(
        c64tape.parse_tap(formats.turbo_prg_to_tap(prg, fmt)), SR)
    tap = wav2tapsp.wav_to_tap(samples, SR)
    assert formats.turbo_tap_to_prg(tap, fmt) == prg
