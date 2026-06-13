# Copyright 2022 Josh Bailey (josh@vandervecken.com)

"""C64 fast / "turbo" tape loader formats.

The C64 ROM loader (see :mod:`wav2tapsp.c64tape`) is slow: it uses three pulse
lengths and two pulses per bit. Commercial and public-domain *fast loaders*
replaced it with simpler, shorter encodings -- almost always **two** pulse
lengths and **one pulse per bit** (a short pulse for 0, a long pulse for 1) --
trading robustness for speed.

For the purpose of testing ``wav2tapsp`` what matters is the *pulse-timing
scheme*: the two pulse widths a loader uses, because those are exactly what the
WAV -> TAP conversion must preserve. This module models that scheme with a
single generic two-level codec parameterised by each loader's real pulse widths
(taken from the references below), rather than re-implementing each loader's
idiosyncratic header/protocol byte-for-byte.

Pulse widths are stored in TAP units (CPU cycles / 8), which is what a .tap file
actually holds; the CPU-cycle value is ``tap * 8``.

References
---------
- TAP file format (pulse byte = CPU cycles / 8; 0x00 = overflow):
  http://www.computerbrains.com/tapformat.html  (verbatim mirror of Per Hakan
  Sundell's original spec; unusedino.de/ec64/technical/formats/tap.html is the
  usual home)
- PAL CPU = 985248 Hz, NTSC = 1022727 Hz:
  https://codebase.c64.org/doku.php?id=base:cpu_clocking
- Standard ROM loader (0x30/0x42/0x56, dipole, odd parity, 10s leader):
  https://web.archive.org/web/20180925092720/http://c64tapes.org/dokuwiki/doku.php?id=loaders:rom_loader
- FinalTAP loader database (per-loader ideal pulse widths and thresholds; the
  de-facto reference, successor to/shared with TAPClean):
  https://github.com/Geert-Jan77/finaltap-console  (main.c ``ft[]`` table and
  ``docs/C64 Tape Formats/``)

Turbo loaders are confirmed to use one pulse per bit, thresholded: a pulse below
the loader's threshold is bit 0 (short), above it is bit 1 (long).
"""

from dataclasses import dataclass

from wav2tapsp import c64tape


@dataclass(frozen=True)
class LoaderFormat:
    """A two-level (one-pulse-per-bit) tape loader pulse scheme.

    short_tap / long_tap : the bit-0 and bit-1 pulse widths, in TAP units.
    pilot_tap            : the leader/pilot pulse width (usually == short_tap).
    pilot_count          : number of pilot pulses written before the data.
    sync_byte            : byte value used to lock bit alignment after the pilot.
    reference            : primary-source URL for the pulse widths.
    note                 : free-form provenance / caveats.
    """

    name: str
    short_tap: int
    long_tap: int
    pilot_count: int = 1000
    sync_byte: int = 0x5A
    pilot_tap: int = None
    reference: str = ''
    note: str = ''

    @property
    def pilot(self):
        return self.short_tap if self.pilot_tap is None else self.pilot_tap

    @property
    def short_cycles(self):
        return self.short_tap * 8

    @property
    def long_cycles(self):
        return self.long_tap * 8

    @property
    def threshold_tap(self):
        """Midpoint used to classify a recovered pulse as 0 (short) or 1 (long)."""
        return (self.short_tap + self.long_tap) / 2.0

    @property
    def gap_cycles(self):
        """Cycle distance between the two pulse lengths (drives timing margin)."""
        return (self.long_tap - self.short_tap) * 8


# --------------------------------------------------------------------------
# Generic two-level turbo encode / decode
# --------------------------------------------------------------------------

def _bits_msb(value, n=8):
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def _emit_byte_bits(pulses, value, fmt):
    for bit in _bits_msb(value):
        pulses.append(fmt.long_tap if bit else fmt.short_tap)


def turbo_prg_to_tap(prg, fmt, trailer=64):
    """Encode a PRG as a .tap using the two-level scheme of ``fmt``.

    Stream layout: pilot pulses, a sync byte, a 4-byte header (load/end address,
    little-endian), the program data, and an XOR checksum -- every byte sent
    MSB-first, one pulse per bit.
    """
    if len(prg) < 2:
        raise ValueError('PRG too short')
    start = prg[0] | (prg[1] << 8)
    data = prg[2:]
    end = start + len(data)
    header = [start & 0xff, (start >> 8) & 0xff, end & 0xff, (end >> 8) & 0xff]

    checksum = 0
    for b in header:
        checksum ^= b
    for b in data:
        checksum ^= b

    pulses = [fmt.pilot] * fmt.pilot_count
    _emit_byte_bits(pulses, fmt.sync_byte, fmt)
    for b in header:
        _emit_byte_bits(pulses, b, fmt)
    for b in data:
        _emit_byte_bits(pulses, b, fmt)
    _emit_byte_bits(pulses, checksum, fmt)
    pulses += [fmt.pilot] * trailer

    return c64tape.wrap_tap(pulses)


def _pulses_to_bits(pulses, fmt):
    thr = fmt.threshold_tap
    bits = []
    for p in pulses:
        if p <= 0 or p >= 256:
            continue            # dropped first pulse / gap -- only ever in pilot
        bits.append(1 if p >= thr else 0)
    return bits


def _read_byte_bits(bits, pos):
    value = 0
    for k in range(8):
        value = (value << 1) | bits[pos + k]
    return value, pos + 8


def turbo_tap_to_prg(tap_bytes, fmt):
    """Decode a two-level .tap produced for ``fmt`` back into a PRG image."""
    bits = _pulses_to_bits(c64tape.parse_tap(tap_bytes), fmt)

    # Slide an 8-bit window (MSB-first) to find the sync byte and lock alignment.
    acc = 0
    pos = None
    for i, bit in enumerate(bits):
        acc = ((acc << 1) | bit) & 0xff
        if i >= 7 and acc == fmt.sync_byte:
            pos = i + 1
            break
    if pos is None:
        raise c64tape.DecodeError('sync byte 0x%02x not found' % fmt.sync_byte)

    def need(n):
        if pos + n > len(bits):
            raise c64tape.DecodeError('truncated turbo stream')

    need(32)
    header = []
    for _ in range(4):
        b, pos = _read_byte_bits(bits, pos)
        header.append(b)
    start = header[0] | (header[1] << 8)
    end = header[2] | (header[3] << 8)
    data_len = end - start
    if data_len < 0 or data_len > 0x10000:
        raise c64tape.DecodeError('implausible data length %d' % data_len)

    need(data_len * 8 + 8)
    data = []
    for _ in range(data_len):
        b, pos = _read_byte_bits(bits, pos)
        data.append(b)
    checksum, pos = _read_byte_bits(bits, pos)

    calc = 0
    for b in header:
        calc ^= b
    for b in data:
        calc ^= b
    if calc != checksum:
        raise c64tape.DecodeError('checksum mismatch: got %d want %d' % (calc, checksum))

    import struct
    return struct.pack('<H', start) + bytes(data)


# --------------------------------------------------------------------------
# Registry of real loader pulse timings (populated from the references above)
# --------------------------------------------------------------------------

# Two-level fast loaders, ordered slowest -> fastest (smaller short/long gap).
# Pulse widths are in TAP units (cycles/8) taken from the FinalTAP ``ft[]``
# database (ideal short = sp, ideal long = lp); the CPU-cycle value is tap*8.
# Only the pulse widths are loader-authentic here -- the framing (pilot, sync
# byte, address header, checksum) is the uniform test harness above.
FORMATS = [
    LoaderFormat(
        name='firebird',
        short_tap=0x44, long_tap=0x7E,
        reference='https://github.com/Geert-Jan77/finaltap-console',
        note='Firebird T1: sp=0x44 (544cyc), lp=0x7E (1008cyc). Slow turbo; widest pulses.'),
    LoaderFormat(
        name='novaload',
        short_tap=0x24, long_tap=0x56,
        reference='https://github.com/Geert-Jan77/finaltap-console',
        note='Novaload: sp=0x24 (288cyc), lp=0x56 (688cyc), threshold 0x3D.'),
    LoaderFormat(
        name='freeload',
        short_tap=0x24, long_tap=0x42,
        reference='https://github.com/Geert-Jan77/finaltap-console',
        note='Freeload (Ocean/US Gold etc): sp=0x24 (288cyc), lp=0x42 (528cyc), sync 0x5A.'),
    LoaderFormat(
        name='turbotape250',
        short_tap=0x1A, long_tap=0x28,
        reference='https://github.com/Geert-Jan77/finaltap-console',
        note='Turbo Tape 250 (Holch): sp=0x1A (208cyc), lp=0x28 (320cyc), threshold 0x20.'),
    LoaderFormat(
        name='turrican',
        short_tap=0x1B, long_tap=0x27,
        reference='https://github.com/Geert-Jan77/finaltap-console',
        note='Turrican: sp=0x1B (216cyc), lp=0x27 (312cyc). Smallest gap -> needs highest sample rate.'),
]


def get(name):
    for fmt in FORMATS:
        if fmt.name == name:
            return fmt
    raise KeyError(name)


# --------------------------------------------------------------------------
# Unified loader interface (lets tests treat ROM + turbo formats the same)
# --------------------------------------------------------------------------

class Loader:
    """A named codec plus the pulse geometry needed for the timing analysis.

    short_cycles : width of the shortest pulse the format uses (CPU cycles).
    gap_cycles   : smallest cycle distance between two adjacent pulse classes
                   (this is what sets the audio sample-rate requirement).
    """

    def __init__(self, name, encode, decode, short_cycles, gap_cycles, reference=''):
        self.name = name
        self.encode = encode
        self.decode = decode
        self.short_cycles = short_cycles
        self.gap_cycles = gap_cycles
        self.reference = reference

    def __repr__(self):
        return 'Loader(%r)' % self.name


def _rom_loader():
    # Standard ROM format: three pulse lengths, two pulses/bit (see c64tape).
    short_cycles = c64tape.SHORT * 8
    gap_cycles = min(c64tape.MEDIUM - c64tape.SHORT, c64tape.LONG - c64tape.MEDIUM) * 8
    return Loader(
        name='rom',
        encode=c64tape.prg_to_tap,
        decode=c64tape.tap_to_prg,
        short_cycles=short_cycles,
        gap_cycles=gap_cycles,
        reference='http://unusedino.de/ec64/technical/formats/tap.html')


def _turbo_loader(fmt):
    return Loader(
        name=fmt.name,
        encode=lambda prg, fmt=fmt: turbo_prg_to_tap(prg, fmt),
        decode=lambda tap, fmt=fmt: turbo_tap_to_prg(tap, fmt),
        short_cycles=fmt.short_cycles,
        gap_cycles=fmt.gap_cycles,
        reference=fmt.reference)


ALL_LOADERS = [_rom_loader()] + [_turbo_loader(f) for f in FORMATS]

