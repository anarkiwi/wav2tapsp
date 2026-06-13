# Copyright 2022 Josh Bailey (josh@vandervecken.com)

"""First-principles limits for digitising C64 tape audio, and helpers to
measure them by round-tripping through ``wav2tapsp``.

wav2tapsp recovers a pulse by counting the samples between consecutive rising
zero crossings and scaling: ``recovered_cycles = delta_samples * cpufreq / SR``.
Two things therefore bound a usable WAV:

1. Sample rate (timing resolution). Two facts:

   * Nyquist. A pulse is one full wave cycle of frequency ``f = cpufreq /
     pulse_cycles``. To represent the *shortest* pulse at all you need
     ``SR >= 2 * cpufreq / short_cycles``; below that the fastest pulses alias
     away and nothing decodes.

   * Class separation. Each crossing is detected at an integer sample, so the
     recovered length carries up to ~``jitter`` samples of error, i.e.
     ``jitter * cpufreq / SR`` cycles. To keep a pulse on the correct side of
     the threshold (which sits ``gap/2`` cycles from each nominal width) we need
     ``jitter * cpufreq / SR < gap_cycles / 2``  ->  ``SR > 2*jitter*cpufreq/gap``.
     With the worst-case ``jitter = 2`` that is ``SR > 4 * cpufreq / gap``.

   The reliable minimum is ``safety * max(`` of the two ``)``. Because ``gap`` is
   usually the binding term, the requirement scales like ``1 / gap`` -- faster
   loaders (smaller gap, shorter pulses) need proportionally higher sample
   rates.

2. Amplitude resolution (bit depth). Zero-crossing detection only needs the
   *sign* of the signal, so for a clean recording one bit (the sign) is enough;
   amplitude resolution barely matters. Sample rate, not bit depth, is the real
   constraint. ``minimum_resolution_bits`` measures this empirically.
"""

import math

from wav2tapsp import c64tape, convert

CPU_FREQ = convert.DEFAULT_CPUFREQ


# --------------------------------------------------------------------------
# Closed-form bounds
# --------------------------------------------------------------------------

def nyquist_floor_hz(short_cycles, cpufreq=CPU_FREQ):
    """Below this the shortest pulse cannot be represented (aliases away)."""
    return 2.0 * cpufreq / short_cycles


def classification_floor_hz(gap_cycles, cpufreq=CPU_FREQ, jitter=2.0):
    """Below this, sample jitter can push a pulse across the decision threshold."""
    return 2.0 * jitter * cpufreq / gap_cycles


def reliable_samplerate_hz(short_cycles, gap_cycles, cpufreq=CPU_FREQ,
                           safety=2.0, jitter=2.0):
    """Lowest sample rate that should decode with margin."""
    floor = max(nyquist_floor_hz(short_cycles, cpufreq),
                classification_floor_hz(gap_cycles, cpufreq, jitter))
    return int(math.ceil(safety * floor))


def aliasing_samplerate_hz(short_cycles, cpufreq=CPU_FREQ):
    """About one sample per shortest pulse: comfortably below Nyquist, so the
    round trip is guaranteed to fail. Used as the lower bracket in tests."""
    return int(cpufreq // short_cycles)


# --------------------------------------------------------------------------
# Empirical round-trip measurement
# --------------------------------------------------------------------------

def roundtrip(loader, prg, samplerate, cpufreq=CPU_FREQ, waveform='square',
              bits=None, amplitude=20000):
    """PRG -> TAP -> WAV samples -> wav2tapsp -> TAP -> PRG, returning the PRG."""
    tap0 = loader.encode(prg)
    pulses = c64tape.parse_tap(tap0)
    samples = c64tape.tap_pulses_to_samples(
        pulses, samplerate, cpufreq=cpufreq, amplitude=amplitude, waveform=waveform)
    if bits is not None:
        samples = c64tape.quantize_resolution(samples, bits, amplitude=amplitude)
    recovered_tap = convert.wav_to_tap(samples, samplerate, cpufreq)
    return loader.decode(recovered_tap)


def roundtrip_ok(loader, prg, samplerate, **kwargs):
    """True iff the round trip recovers the original PRG exactly."""
    try:
        return roundtrip(loader, prg, samplerate, **kwargs) == prg
    except Exception:
        return False


def minimum_samplerate(loader, prg, cpufreq=CPU_FREQ, points=18, lo=None, hi=None,
                       **kwargs):
    """Find the lowest sample rate (on a geometric grid) that round-trips.

    Scans from ``lo`` (default: the aliasing floor) up to ``hi`` (default: the
    reliable estimate) and returns the first sample rate that decodes exactly,
    or None if none in range does.
    """
    if lo is None:
        lo = aliasing_samplerate_hz(loader.short_cycles, cpufreq)
    if hi is None:
        hi = reliable_samplerate_hz(loader.short_cycles, loader.gap_cycles, cpufreq)
    ratio = (hi / lo) ** (1.0 / (points - 1))
    for k in range(points):
        sr = int(lo * (ratio ** k))
        if roundtrip_ok(loader, prg, sr, cpufreq=cpufreq, **kwargs):
            return sr
    return None


def minimum_resolution_bits(loader, prg, samplerate, cpufreq=CPU_FREQ,
                            waveform='sine', max_bits=16, **kwargs):
    """Find the smallest amplitude resolution (in bits) that still round-trips
    at ``samplerate``. Scans 1, 2, ... up to ``max_bits``."""
    for bits in range(1, max_bits + 1):
        if roundtrip_ok(loader, prg, samplerate, cpufreq=cpufreq,
                        waveform=waveform, bits=bits, **kwargs):
            return bits
    return None


def measure_limits(loaders, prg, cpufreq=CPU_FREQ):
    """Return a per-loader table of theoretical and measured limits."""
    rows = []
    for loader in loaders:
        srel = reliable_samplerate_hz(loader.short_cycles, loader.gap_cycles, cpufreq)
        rows.append({
            'name': loader.name,
            'short_cycles': loader.short_cycles,
            'gap_cycles': loader.gap_cycles,
            'nyquist_hz': int(nyquist_floor_hz(loader.short_cycles, cpufreq)),
            'classification_hz': int(classification_floor_hz(loader.gap_cycles, cpufreq)),
            'reliable_hz': srel,
            'measured_min_hz': minimum_samplerate(loader, prg, cpufreq),
            'measured_min_bits': minimum_resolution_bits(loader, prg, 2 * srel, cpufreq),
        })
    return rows
