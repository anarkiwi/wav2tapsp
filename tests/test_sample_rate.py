"""Minimum reliable WAV sample rate per loader, derived from first principles.

For each loader we bracket the real minimum between two closed-form bounds and
confirm the bracket empirically by round-tripping through wav2tapsp:

  * at the *aliasing* rate (~1 sample per shortest pulse, below Nyquist) the
    round trip MUST fail -- the pulses are not representable; and
  * at the *reliable* rate (safety x max(Nyquist, class-separation)) it MUST
    succeed.

See wav2tapsp.analysis for the derivation. The headline result is that the
requirement scales like 1/gap: the smaller a loader's short/long pulse gap, the
higher the sample rate it needs.
"""

import pytest

from wav2tapsp import analysis, c64tape, formats

LOADERS = formats.ALL_LOADERS
LOADER_IDS = [loader.name for loader in LOADERS]


def _prg():
    return c64tape.sample_basic_prg()


@pytest.mark.parametrize('loader', LOADERS, ids=LOADER_IDS)
def test_reliable_rate_decodes(loader):
    sr = analysis.reliable_samplerate_hz(loader.short_cycles, loader.gap_cycles)
    assert analysis.roundtrip_ok(loader, _prg(), sr)


@pytest.mark.parametrize('loader', LOADERS, ids=LOADER_IDS)
def test_below_nyquist_fails(loader):
    sr = analysis.aliasing_samplerate_hz(loader.short_cycles)
    assert not analysis.roundtrip_ok(loader, _prg(), sr)


@pytest.mark.parametrize('loader', LOADERS, ids=LOADER_IDS)
def test_measured_minimum_is_bracketed(loader):
    measured = analysis.minimum_samplerate(loader, _prg())
    alias = analysis.aliasing_samplerate_hz(loader.short_cycles)
    reliable = analysis.reliable_samplerate_hz(loader.short_cycles, loader.gap_cycles)
    assert measured is not None
    # the real minimum sits above the aliasing floor and at/below the reliable
    # estimate -- i.e. the first-principles bounds bracket reality.
    assert alias < measured <= reliable


def test_required_rate_scales_inversely_with_gap():
    # Smaller short/long pulse gap -> higher required sample rate.
    ordered = sorted(LOADERS, key=lambda loader: loader.gap_cycles)
    rates = [analysis.reliable_samplerate_hz(loader.short_cycles, loader.gap_cycles)
             for loader in ordered]
    # strictly decreasing as the gap widens
    assert all(a > b for a, b in zip(rates, rates[1:])), rates


def test_classification_floor_is_the_binding_constraint():
    # For every loader here, separating the two pulse widths (not Nyquist) is
    # what sets the requirement -- the basis of the 1/gap scaling.
    for loader in LOADERS:
        nyq = analysis.nyquist_floor_hz(loader.short_cycles)
        cls = analysis.classification_floor_hz(loader.gap_cycles)
        assert cls > nyq, loader.name
