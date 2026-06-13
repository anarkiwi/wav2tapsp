# wav2tapsp
wav2tapsp

[![CI](https://github.com/anarkiwi/wav2tapsp/actions/workflows/ci.yml/badge.svg)](https://github.com/anarkiwi/wav2tapsp/actions/workflows/ci.yml)

Converts WAV files (recordings of C64 cassette tapes) to C64 TAP files an emulator can read.

Hardly unique, but also somewhat easier to experiment with than having to hack ancient C code...

```
$ pip3 install -r requirements.txt
$ ./wav2tapsp.py
usage: wav2tapsp.py [-h] [--cpufreq CPUFREQ] wavfile
wav2tapsp.py: error: the following arguments are required: wavfile
```

## Testing

Tests run automatically in CI (GitHub Actions) across Python 3.10 - 3.13. To run
them yourself:

```
$ pip3 install -r requirements-dev.txt
$ pytest -v
```

The headline test is a full, emulator-free round trip that proves `wav2tapsp`
preserves tape timing well enough to recover the original program:

```
PRG  ->  TAP  ->  WAV  --(wav2tapsp)-->  TAP  ->  PRG
```

It generates a real C64 BASIC program, encodes it as a C64 ROM-format `.tap`
(leader / countdown sync / per-byte dipole markers / odd parity / per-block
checksum, two block copies), synthesises a `.wav` of the tape audio, runs it back
through `wav2tapsp`, decodes the resulting `.tap`, and asserts the recovered PRG
is byte-identical to the original *and* still a structurally valid, runnable
BASIC program. The C64 tape format helpers live in [`c64tape.py`](c64tape.py).
