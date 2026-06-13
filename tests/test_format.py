"""Unit tests for the C64 tape toolkit (no audio involved)."""

import struct

import numpy as np
import pytest

from wav2tapsp import c64tape


def test_sample_prg_is_valid_basic():
    prg = c64tape.sample_basic_prg()
    # load address is BASIC start ($0801)
    assert prg[:2] == struct.pack('<H', c64tape.BASIC_START)
    lines = c64tape.parse_basic_prg(prg)
    assert [ln for ln, _ in lines] == [10, 20]
    # line 10 is PRINT "HELLO, WORLD!"
    assert lines[0][1] == bytes([c64tape.TOK_PRINT]) + b'"HELLO, WORLD!"'


def test_parse_basic_prg_rejects_bad_link():
    prg = bytearray(c64tape.sample_basic_prg())
    prg[2] ^= 0xff  # corrupt the first line's link pointer
    with pytest.raises(ValueError):
        c64tape.parse_basic_prg(bytes(prg))


def test_tap_container_roundtrip():
    prg = c64tape.sample_basic_prg()
    tap = c64tape.prg_to_tap(prg)
    assert tap[:12] == c64tape.TAP_MAGIC
    pulses = c64tape.parse_tap(tap)
    # every pulse is a legal single-byte length
    assert pulses
    assert all(0 < p < 256 for p in pulses)
    # only short/medium/long lengths are emitted
    assert set(pulses) <= {c64tape.SHORT, c64tape.MEDIUM, c64tape.LONG}


def test_encode_then_decode_without_audio():
    # The encoder and decoder are self-consistent end to end.
    prg = c64tape.sample_basic_prg()
    tap = c64tape.prg_to_tap(prg)
    assert c64tape.tap_to_prg(tap) == prg


def test_decode_recovers_header_addresses():
    prg = c64tape.sample_basic_prg()
    tap = c64tape.prg_to_tap(prg)
    pulses = c64tape.parse_tap(tap)
    blocks = []
    syms = c64tape._classify(pulses)
    i = 0
    while True:
        i = c64tape._next_marker(syms, i)
        if i >= len(syms):
            break
        block, i = c64tape._read_block(syms, i)
        if block:
            blocks.append(block)
    # two header copies + two data copies
    assert len(blocks) == 4
    header = c64tape._strip_block(blocks[0])
    start = header[1] | (header[2] << 8)
    end = header[3] | (header[4] << 8)
    assert start == c64tape.BASIC_START
    assert end == c64tape.BASIC_START + (len(prg) - 2)


def test_corrupt_pulse_is_detected():
    prg = c64tape.sample_basic_prg()
    pulses = c64tape.parse_tap(c64tape.prg_to_tap(prg))
    # flip a data pulse in the middle of the stream into the wrong class
    mid = len(pulses) // 2
    pulses[mid] = c64tape.LONG if pulses[mid] != c64tape.LONG else c64tape.SHORT
    with pytest.raises(c64tape.DecodeError):
        c64tape.pulses_to_prg(pulses)


def test_arbitrary_prg_roundtrips_without_audio():
    # Not a BASIC program -- arbitrary bytes with a load address still survive.
    prg = struct.pack('<H', 0xC000) + bytes(range(200))
    tap = c64tape.prg_to_tap(prg, file_type=3)
    assert c64tape.tap_to_prg(tap) == prg


def test_tap_pulses_to_samples_shape():
    pulses = [c64tape.SHORT, c64tape.MEDIUM, c64tape.LONG] * 10
    samples = c64tape.tap_pulses_to_samples(pulses, 48000)
    assert samples.dtype == np.int16
    # only +/- amplitude values are produced (clean square wave)
    assert set(np.unique(samples).tolist()) <= {-20000, 20000}
