# wav2tapsp

[![CI](https://github.com/anarkiwi/wav2tapsp/actions/workflows/ci.yml/badge.svg)](https://github.com/anarkiwi/wav2tapsp/actions/workflows/ci.yml)

Converts WAV files (recordings of Commodore 64 cassette tapes) into C64 `.tap`
files an emulator can read.

Hardly unique, but also somewhat easier to experiment with than having to hack
ancient C code...

## Install

```
$ pip install .
```

or, once released, `pip install wav2tapsp`.

## Command line

```
$ wav2tapsp recording.wav        # writes recording.tap
$ wav2tapsp --cpufreq 1022727 recording.wav   # NTSC
```

(`python -m wav2tapsp recording.wav` works too.)

## Library

```python
import scipy.io.wavfile
import wav2tapsp

sr, samples = scipy.io.wavfile.read('recording.wav')
tap_bytes = wav2tapsp.wav_to_tap(samples, sr)          # bytes of a .tap file
pulses = wav2tapsp.wav_to_pulses(samples, sr)          # raw TAP pulse lengths
```

It works by finding rising zero crossings in the audio and emitting the time
between them as TAP pulse lengths (CPU cycles / 8).

## C64 tape formats

The package can synthesise and decode real C64 tape formats end to end (used by
the tests, and handy for generating test tapes):

- **Standard ROM / Kernal loader** — three pulse lengths, two pulses per bit,
  odd parity, per-block checksum, two block copies.
  [`c64tape.py`](src/wav2tapsp/c64tape.py).
- **Fast / turbo loaders** — two pulse lengths, one pulse per bit (a short pulse
  for 0, a long pulse for 1). [`formats.py`](src/wav2tapsp/formats.py) models the
  *pulse-timing scheme* of several real loaders using their documented pulse
  widths:

  | Loader | short / long pulse (TAP, cycles) | reference |
  | --- | --- | --- |
  | Firebird T1 | `$44` / `$7E` (544 / 1008) | FinalTAP `ft[]` |
  | Novaload | `$24` / `$56` (288 / 688) | FinalTAP docs |
  | Freeload | `$24` / `$42` (288 / 528) | FinalTAP docs |
  | Turbo Tape 250 | `$1A` / `$28` (208 / 320) | FinalTAP docs / [c64-wiki](https://www.c64-wiki.com/wiki/Turbo_Tape) |
  | Turrican | `$1B` / `$27` (216 / 312) | FinalTAP `ft[]` |

References: the [TAP format](http://www.computerbrains.com/tapformat.html)
(pulse byte = CPU cycles / 8), [PAL/NTSC CPU clocks](https://codebase.c64.org/doku.php?id=base:cpu_clocking)
(985248 / 1022727 Hz), the [ROM loader timing](https://web.archive.org/web/20180925092720/http://c64tapes.org/dokuwiki/doku.php?id=loaders:rom_loader)
(`$30`/`$42`/`$56`), and the [FinalTAP loader database](https://github.com/Geert-Jan77/finaltap-console)
(per-loader pulse widths, the de-facto reference).

## Testing

Tests run in CI across Python 3.10–3.13. To run them yourself:

```
$ pip install -e .[test]
$ pytest -v
```

The headline test is a full, emulator-free round trip that proves `wav2tapsp`
preserves tape timing well enough to recover the original program:

```
PRG  ->  TAP  ->  WAV  --(wav2tapsp)-->  TAP  ->  PRG
```

A real C64 BASIC program is generated, encoded as a `.tap`, rendered to a `.wav`,
run back through `wav2tapsp`, and decoded; the recovered PRG must be
byte-identical to the original *and* still a structurally valid, runnable BASIC
program. The same round trip runs for every fast-loader format above.

### Minimum sample rate and resolution, from first principles

`wav2tapsp` recovers a pulse by counting samples between rising zero crossings:
`recovered_cycles = Δsamples × cpufreq / SR`. Two bounds follow (derived in
[`analysis.py`](src/wav2tapsp/analysis.py), checked per format in the tests):

- **Sample rate.** A pulse is one wave cycle of frequency `cpufreq / pulse`, so
  representing the shortest pulse needs `SR ≥ 2·cpufreq/short` (Nyquist). To keep
  a pulse on the correct side of the short/long threshold despite ±2 samples of
  crossing jitter needs `SR > 4·cpufreq/gap`, where `gap` is the cycle distance
  between the two pulse widths. The reliable minimum is `2 × max(...)`. Because
  `gap` dominates, **the requirement scales like 1/gap** — faster loaders need
  proportionally higher sample rates (PAL, safety ×2):

  | Loader | gap (cycles) | reliable min sample rate |
  | --- | --- | --- |
  | Firebird T1 | 464 | ~17.0 kHz |
  | Novaload | 400 | ~19.7 kHz |
  | Freeload | 240 | ~32.8 kHz |
  | C64 ROM | 144 | ~54.7 kHz |
  | Turbo Tape 250 | 112 | ~70.4 kHz |
  | Turrican | 96 | ~82.1 kHz |

  The tests confirm each loader fails below Nyquist and decodes at its reliable
  rate, and that the measured minimum is bracketed between the two.

- **Resolution (bit depth).** Zero-crossing detection depends only on the *sign*
  of the signal, so **one bit (the sign) is enough** for a clean recording;
  amplitude resolution barely matters. The tests verify the round trip survives
  1-bit quantisation of a sine wave for every loader. Sample rate, not bit depth,
  is the real constraint.

## Layout

- [`src/wav2tapsp/convert.py`](src/wav2tapsp/convert.py) — the WAV → TAP converter (the tool).
- [`src/wav2tapsp/c64tape.py`](src/wav2tapsp/c64tape.py) — standard ROM tape codec, PRG/BASIC helpers, WAV synthesis.
- [`src/wav2tapsp/formats.py`](src/wav2tapsp/formats.py) — fast/turbo loader registry and codec.
- [`src/wav2tapsp/analysis.py`](src/wav2tapsp/analysis.py) — first-principles sample-rate/resolution bounds.
