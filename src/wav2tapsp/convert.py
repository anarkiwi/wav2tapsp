# Copyright 2022 Josh Bailey (josh@vandervecken.com)

"""scipy based WAV -> C64 TAP converter.

https://web.archive.org/web/20180709173001/http://c64tapes.org/dokuwiki/doku.php?id=analyzing_loaders#tap_format
http://unusedino.de/ec64/technical/formats/tap.html
"""

import struct

import numpy as np
import pandas as pd
import scipy.io.wavfile

# PAL CPU frequency in Hz.
DEFAULT_CPUFREQ = 985248


def wav_to_pulses(x, sr, cpufreq=DEFAULT_CPUFREQ):
    """Convert PCM samples (as read by scipy.io.wavfile) into TAP pulse bytes.

    Returns a uint8 numpy array of TAP pulse lengths (CPU cycles / 8), one per
    rising zero crossing in the signal.
    """
    # TODO: could add audio/gain processing here.
    if len(x.shape) == 1:
        df = pd.DataFrame(x, columns=['s'], dtype=pd.Float32Dtype())
    else:
        # if stereo, mean of channel samples
        df = pd.DataFrame(np.column_stack(np.transpose(x)), columns=['x', 'y'], dtype=pd.Float32Dtype())
        df['s'] = df.mean(axis=1)

    # calculate timestamps
    df['t'] = df.index / sr * 1e6
    df['t'] = df['t'].astype(np.uint64)

    # calculate zero crossings, between samples
    df['ls'] = df['s'].shift(1).fillna(0)
    df.loc[(df.ls == df.s), ['ls', 's']] = pd.NA
    df['zo'] = df.ls / (df.ls - df.s)
    df.loc[~((df.ls < 0) & (df.s >= 0)), ['zo']] = pd.NA
    df['lastzo'] = df['zo'].shift().ffill()
    df['zt'] = df['t'] + df['zo'] - df['lastzo']

    # select zero crossings
    df = df[df['zt'].notna()]
    # calculate pulse length bytes
    df['pl'] = df.zt.diff().fillna(0) * ((1e-6 * cpufreq) / 8)
    # remove overflows
    df.loc[df['pl'] > 255, ['pl']] = 0
    df['pl'] = df.pl.fillna(0).astype(np.uint8)
    return df['pl'].to_numpy()


def pulses_to_tap(pulses):
    """Wrap an array of TAP pulse bytes in a C64-TAPE-RAW (version 0) container."""
    out = bytearray()
    out += bytes('C64-TAPE-RAW', 'utf8')
    # version 0, no long pulse handling
    out += struct.pack('b', 0)
    # reserved
    out += bytes(3)
    # payload
    out += struct.pack('I', len(pulses))
    out += pulses.tobytes()
    return bytes(out)


def wav_to_tap(x, sr, cpufreq=DEFAULT_CPUFREQ):
    """Convert PCM samples into a complete .tap file as bytes."""
    return pulses_to_tap(wav_to_pulses(x, sr, cpufreq))


def read_wav_to_tap(wavfile, cpufreq=DEFAULT_CPUFREQ):
    """Read a WAV file from disk and return its .tap bytes."""
    sr, x = scipy.io.wavfile.read(wavfile)
    return wav_to_tap(x, sr, cpufreq)
