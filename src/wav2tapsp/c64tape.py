# Copyright 2022 Josh Bailey (josh@vandervecken.com)

"""Self-contained Commodore 64 cassette-tape toolkit (standard ROM format).

Implements just enough of the C64 ROM ("Kernal") tape format to drive a full
round-trip test of ``wav2tapsp`` with no emulator or copyrighted ROMs:

    PRG  ->  TAP  ->  WAV  --(wav2tapsp)-->  TAP  ->  PRG

The format implemented here (leader / countdown sync / per-byte dipole markers /
odd parity / per-block checksum / two block copies) is the standard format the
C64 ROM SAVEs and LOADs, so the produced .tap files are real C64 tapes.

This module also provides the shared building blocks used by ``formats`` (the
fast-loader registry): the TAP container, the WAV synthesiser and the BASIC PRG
helpers.

References:
  http://unusedino.de/ec64/technical/formats/tap.html
  https://web.archive.org/web/20180709173001/http://c64tapes.org/dokuwiki/doku.php?id=analyzing_loaders
"""

import struct
import wave

import numpy as np

# PAL CPU frequency in Hz (matches wav2tapsp's default).
CPU_FREQ = 985248

# Standard ROM-loader pulse lengths, in TAP units (CPU cycles / 8).
SHORT = 0x30    # 48
MEDIUM = 0x42   # 66
LONG = 0x56     # 86

# Decode thresholds: midpoints between the nominal pulse lengths. A recovered
# pulse below _SM is "short", below _ML is "medium", otherwise "long".
_SM = (SHORT + MEDIUM) // 2     # 57
_ML = (MEDIUM + LONG) // 2      # 76

TAP_MAGIC = b'C64-TAPE-RAW'

# C64 BASIC starts at $0801.
BASIC_START = 0x0801

# A handful of BASIC V2 tokens (enough for the sample program).
TOK_GOTO = 0x89
TOK_PRINT = 0x99

# Countdown "sync" bytes written before each block. The high bit distinguishes
# the first copy of a block (0x89..0x81) from the repeated copy (0x09..0x01).
_SYNC_FIRST = [0x89, 0x88, 0x87, 0x86, 0x85, 0x84, 0x83, 0x82, 0x81]
_SYNC_REPEAT = [0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]

# Tape header (cassette buffer) is always 192 bytes.
HEADER_SIZE = 192


class DecodeError(Exception):
    """Raised when a pulse stream cannot be decoded as a C64 tape."""


# --------------------------------------------------------------------------
# BASIC PRG generation / validation
# --------------------------------------------------------------------------

def make_basic_prg(lines):
    """Build a tokenised C64 BASIC .prg from ``[(line_number, token_bytes), ...]``.

    Returns the full PRG image: 2-byte little-endian load address followed by
    the linked list of BASIC lines and a terminating null link.
    """
    body = bytearray()
    addr = BASIC_START
    for line_no, tokens in lines:
        tokens = bytes(tokens)
        # line = link(2) + line number(2) + tokens + end-of-line(0x00)
        line_len = 2 + 2 + len(tokens) + 1
        next_addr = addr + line_len
        body += struct.pack('<H', next_addr)
        body += struct.pack('<H', line_no)
        body += tokens
        body += b'\x00'
        addr = next_addr
    # null link terminates the program
    body += b'\x00\x00'
    return struct.pack('<H', BASIC_START) + bytes(body)


def sample_basic_prg():
    """A small but genuine, runnable C64 BASIC program."""
    line10 = bytes([TOK_PRINT]) + b'"HELLO, WORLD!"'
    line20 = bytes([TOK_GOTO]) + b' 10'
    return make_basic_prg([(10, line10), (20, line20)])


def parse_basic_prg(prg):
    """Walk a tokenised BASIC PRG, returning ``[(line_number, token_bytes), ...]``.

    Validates the line-link structure (each line's link must point at the next
    line, the program must end with a null link). Raises ``ValueError`` if the
    bytes are not a structurally valid BASIC program.
    """
    if len(prg) < 2:
        raise ValueError('PRG too short')
    load = prg[0] | (prg[1] << 8)
    mem = prg[2:]
    lines = []
    pos = 0
    addr = load
    while True:
        if pos + 2 > len(mem):
            raise ValueError('truncated link pointer')
        link = mem[pos] | (mem[pos + 1] << 8)
        if link == 0:
            break
        if pos + 4 > len(mem):
            raise ValueError('truncated line header')
        line_no = mem[pos + 2] | (mem[pos + 3] << 8)
        k = pos + 4
        while k < len(mem) and mem[k] != 0:
            k += 1
        if k >= len(mem):
            raise ValueError('unterminated BASIC line')
        tokens = bytes(mem[pos + 4:k])
        expected_link = addr + (k + 1 - pos)
        if link != expected_link:
            raise ValueError('bad line link: got $%04x want $%04x' % (link, expected_link))
        lines.append((line_no, tokens))
        addr = link
        pos = k + 1
    return lines


# --------------------------------------------------------------------------
# TAP container (shared with the fast-loader formats)
# --------------------------------------------------------------------------

def wrap_tap(pulses):
    """Wrap a list/array of TAP pulse bytes in a C64-TAPE-RAW (version 0) file."""
    body = bytes(bytearray(pulses))
    out = bytearray()
    out += TAP_MAGIC
    out += bytes([0])           # version 0
    out += bytes(3)             # reserved
    out += struct.pack('<I', len(body))
    out += body
    return bytes(out)


def parse_tap(tap_bytes):
    """Read a version-0 .tap container, returning the list of pulse bytes."""
    if tap_bytes[:12] != TAP_MAGIC:
        raise DecodeError('not a C64 TAP file')
    (length,) = struct.unpack('<I', tap_bytes[16:20])
    return list(tap_bytes[20:20 + length])


# --------------------------------------------------------------------------
# TAP -> WAV (synthesise audio)
# --------------------------------------------------------------------------

def tap_pulses_to_samples(pulses, samplerate, cpufreq=CPU_FREQ, amplitude=20000,
                          waveform='square'):
    """Render TAP pulses as an int16 sample array.

    Each pulse becomes one full wave cycle whose period is ``pulse * 8 / cpufreq``
    seconds, so every pulse boundary is a single rising zero crossing -- exactly
    what wav2tapsp measures.

    waveform='square' produces a hard square wave (isolates timing, used for the
    sample-rate study). waveform='sine' produces one sine cycle per pulse (a more
    realistic model of tape audio, used for the resolution study).
    """
    p = np.asarray(pulses, dtype=np.float64)
    dur_n = p * 8.0 / cpufreq * samplerate     # samples per pulse (float)
    starts = np.empty(len(p) + 1)
    starts[0] = 0.0
    np.cumsum(dur_n, out=starts[1:])
    total = int(np.ceil(starts[-1]))
    n = np.arange(total)
    idx = np.searchsorted(starts, n, side='right') - 1
    np.clip(idx, 0, len(p) - 1, out=idx)
    within = n - starts[idx]
    phase = within / dur_n[idx]                # 0..1 within the pulse
    if waveform == 'square':
        samples = np.where(phase < 0.5, amplitude, -amplitude)
    elif waveform == 'sine':
        samples = amplitude * np.sin(2.0 * np.pi * phase)
    else:
        raise ValueError('unknown waveform %r' % waveform)
    return np.round(samples).astype(np.int16)


def quantize_resolution(samples, bits, amplitude=20000):
    """Requantise a signed sample array to ``bits`` bits of amplitude resolution.

    Uses a mid-rise quantiser with no exact zero level, so ``bits == 1`` reduces
    to the sign of the signal (the coarsest representation that still preserves
    zero crossings). Returns an int16 array.
    """
    if bits < 1:
        raise ValueError('bits must be >= 1')
    step = 2.0 * amplitude / (2 ** bits)
    q = (np.floor(np.asarray(samples, dtype=np.float64) / step) + 0.5) * step
    q = np.clip(q, -amplitude, amplitude)
    return np.round(q).astype(np.int16)


def write_wav(path, samples, samplerate):
    """Write a mono 16-bit PCM WAV file (readable by scipy.io.wavfile)."""
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(np.asarray(samples, dtype='<i2').tobytes())


# --------------------------------------------------------------------------
# PRG -> TAP (encode, standard ROM format)
# --------------------------------------------------------------------------

def _emit_bit(pulses, bit):
    # 0 = short,medium ; 1 = medium,short
    if bit:
        pulses.append(MEDIUM)
        pulses.append(SHORT)
    else:
        pulses.append(SHORT)
        pulses.append(MEDIUM)


def _emit_byte(pulses, value):
    # byte marker dipole: long,medium
    pulses.append(LONG)
    pulses.append(MEDIUM)
    parity = 1
    for i in range(8):
        bit = (value >> i) & 1   # LSB first
        _emit_bit(pulses, bit)
        parity ^= bit
    _emit_bit(pulses, parity)    # odd parity check bit


def _emit_block(pulses, payload, first_copy, leader):
    pulses.extend([SHORT] * leader)
    sync = _SYNC_FIRST if first_copy else _SYNC_REPEAT
    checksum = 0
    for b in payload:
        checksum ^= b
    for b in sync:
        _emit_byte(pulses, b)
    for b in payload:
        _emit_byte(pulses, b)
    _emit_byte(pulses, checksum)


def _make_header(file_type, start, end, name):
    buf = bytearray([0x20] * HEADER_SIZE)
    buf[0] = file_type
    buf[1] = start & 0xff
    buf[2] = (start >> 8) & 0xff
    buf[3] = end & 0xff
    buf[4] = (end >> 8) & 0xff
    nm = name.encode('ascii')[:16]
    buf[5:5 + len(nm)] = nm
    return bytes(buf)


def prg_to_tap(prg, name='WAV2TAPSP', file_type=1, copies=2,
               first_leader=1500, mid_leader=400, trailer=400):
    """Encode a PRG as a C64 ROM-format .tap file (returns bytes).

    A header block (load/end address + name) and the program data block are each
    written ``copies`` times, exactly as the C64 ROM SAVE routine does.
    """
    if len(prg) < 2:
        raise ValueError('PRG too short')
    start = prg[0] | (prg[1] << 8)
    data = prg[2:]
    end = start + len(data)
    header = _make_header(file_type, start, end, name)

    pulses = []
    for copy in range(copies):
        leader = first_leader if copy == 0 else mid_leader
        _emit_block(pulses, header, first_copy=(copy == 0), leader=leader)
    for copy in range(copies):
        _emit_block(pulses, data, first_copy=(copy == 0), leader=mid_leader)
    pulses.extend([SHORT] * trailer)

    return wrap_tap(pulses)


# --------------------------------------------------------------------------
# TAP -> PRG (decode, standard ROM format)
# --------------------------------------------------------------------------

def _classify(pulses):
    out = []
    for v in pulses:
        if v <= 0 or v >= 256:
            out.append('X')      # dropped / overflow / gap
        elif v < _SM:
            out.append('S')
        elif v < _ML:
            out.append('M')
        else:
            out.append('L')
    return out


def _next_marker(syms, i):
    while i < len(syms) and syms[i] != 'L':
        i += 1
    return i


def _read_bit(syms, j):
    a, b = syms[j], syms[j + 1]
    if a == 'S' and b == 'M':
        return 0
    if a == 'M' and b == 'S':
        return 1
    raise DecodeError('bad bit dipole %s%s at %d' % (a, b, j))


def _read_byte(syms, i):
    if not (i + 20 <= len(syms) and syms[i] == 'L' and syms[i + 1] == 'M'):
        raise DecodeError('expected byte marker at %d' % i)
    value = 0
    parity = 1
    j = i + 2
    for b in range(8):
        bit = _read_bit(syms, j)
        value |= bit << b
        parity ^= bit
        j += 2
    if _read_bit(syms, j) != parity:
        raise DecodeError('parity error in byte at %d' % i)
    return value, i + 20


def _read_block(syms, i):
    """Read consecutive bytes (no leader between them) starting at a marker."""
    out = []
    while i + 20 <= len(syms) and syms[i] == 'L' and syms[i + 1] == 'M':
        value, i = _read_byte(syms, i)
        out.append(value)
    return out, i


def _strip_block(block):
    """Split a decoded block into its payload, validating the checksum.

    Layout: 9 countdown sync bytes, then the payload, then a 1-byte XOR checksum.
    """
    if len(block) < 10:
        raise DecodeError('block too short: %d bytes' % len(block))
    payload = block[9:-1]
    checksum = block[-1]
    calc = 0
    for b in payload:
        calc ^= b
    if calc != checksum:
        raise DecodeError('checksum mismatch: got %d want %d' % (calc, checksum))
    return payload


def pulses_to_prg(pulses):
    """Decode a list of TAP pulse bytes back into a PRG image."""
    syms = _classify(pulses)
    blocks = []
    i = 0
    while True:
        i = _next_marker(syms, i)
        if i >= len(syms):
            break
        block, nxt = _read_block(syms, i)
        if block:
            blocks.append(block)
            i = nxt
        else:
            # a lone 'L' that isn't a byte marker (only happens on garbled /
            # undersampled input); skip it so we always make progress.
            i += 1

    if len(blocks) < 2:
        raise DecodeError('expected at least a header and a data block, got %d' % len(blocks))

    header = _strip_block(blocks[0])
    if len(header) != HEADER_SIZE:
        raise DecodeError('header block is %d bytes, expected %d' % (len(header), HEADER_SIZE))
    start = header[1] | (header[2] << 8)
    end = header[3] | (header[4] << 8)
    data_len = end - start

    data = None
    for blk in blocks[1:]:
        payload = _strip_block(blk)
        if payload == header:
            continue            # a repeated copy of the header
        if len(payload) == data_len:
            data = payload
            break
    if data is None:
        raise DecodeError('no data block of length %d found' % data_len)

    return struct.pack('<H', start) + bytes(data)


def tap_to_prg(tap_bytes):
    """Decode a .tap file (bytes) back into a PRG image."""
    return pulses_to_prg(parse_tap(tap_bytes))
