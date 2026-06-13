"""Minimum reliable WAV amplitude resolution (bit depth), from first principles.

wav2tapsp recovers pulses from *zero crossings*, which depend only on the sign
of the signal. So for a clean recording the coarsest possible representation --
one bit, the sign -- already preserves every crossing, and amplitude resolution
is essentially irrelevant. Sample rate (test_sample_rate.py), not bit depth, is
the binding constraint.

These tests use a sine wave (one cycle per pulse, a more realistic model of tape
audio than a hard square) so that bit depth actually affects the samples near
each crossing, then show the round trip still survives down to 1 bit.
"""

import numpy as np
import pytest

from wav2tapsp import analysis, c64tape, formats

LOADERS = formats.ALL_LOADERS
LOADER_IDS = [loader.name for loader in LOADERS]


def _prg():
    return c64tape.sample_basic_prg()


def _comfortable_rate(loader):
    # well above the sample-rate floor so resolution is the only variable
    return 2 * analysis.reliable_samplerate_hz(loader.short_cycles, loader.gap_cycles)


@pytest.mark.parametrize('loader', LOADERS, ids=LOADER_IDS)
def test_one_bit_sign_is_enough(loader):
    # 1-bit quantisation of the sine keeps only its sign, yet the crossings --
    # and therefore the pulse lengths -- are recovered exactly.
    sr = _comfortable_rate(loader)
    assert analysis.roundtrip_ok(loader, _prg(), sr, waveform='sine', bits=1)


@pytest.mark.parametrize('loader', LOADERS, ids=LOADER_IDS)
def test_minimum_resolution_is_one_bit(loader):
    sr = _comfortable_rate(loader)
    bits = analysis.minimum_resolution_bits(loader, _prg(), sr, waveform='sine')
    assert bits == 1


@pytest.mark.parametrize('bits', [1, 2, 4, 8, 16])
def test_full_resolution_range_round_trips(bits):
    # Across the whole practical bit-depth range, a comfortable sample rate
    # round-trips -- resolution buys nothing here.
    loader = next(loader for loader in LOADERS if loader.name == 'rom')
    sr = _comfortable_rate(loader)
    assert analysis.roundtrip_ok(loader, _prg(), sr, waveform='sine', bits=bits)


def test_quantiser_one_bit_is_pure_sign():
    # sanity-check the resolution helper: 1 bit collapses to +/- (the sign).
    x = c64tape.tap_pulses_to_samples([c64tape.MEDIUM] * 20, 96000, waveform='sine')
    q = c64tape.quantize_resolution(x, bits=1, amplitude=20000)
    levels = set(np.unique(q).tolist())
    assert len(levels) == 2
    assert all(v < 0 or v > 0 for v in levels)  # no zero level; pure sign
