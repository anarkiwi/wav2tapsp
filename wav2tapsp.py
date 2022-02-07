#!/usr/bin/python

# Copyright 2022 Josh Bailey (josh@vandervecken.com)

# scipy based WAV -> C64 TAP converter.

# https://web.archive.org/web/20180709173001/http://c64tapes.org/dokuwiki/doku.php?id=analyzing_loaders#tap_format
# http://unusedino.de/ec64/technical/formats/tap.html

import argparse
import struct
import pandas as pd
import numpy as np
import scipy.io.wavfile

parser = argparse.ArgumentParser(description='Convert WAV file into a C64 .tap file')
parser.add_argument('wavfile', help='input WAV file')
parser.add_argument('--cpufreq', default=int(985248), help='CPU frequency in Hz')
args = parser.parse_args()

# TODO: could add audio/gain processing here.
sr, x = scipy.io.wavfile.read(args.wavfile)
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
df['lastzo'] = df['zo'].shift().fillna(method='ffill')
df['zt'] = df['t'] + df['zo'] - df['lastzo']

# select zero crossings
df = df[df['zt'].notna()]
# calculate pulse length bytes
df['pl'] = df.zt.diff().fillna(0) * ((1e-6 * args.cpufreq) / 8)
# remove overflows
df.loc[df['pl'] > 255, ['pl']] = 0
df['pl'] = df.pl.fillna(0).astype(np.uint8)

outfile = args.wavfile.replace('.wav', '.tap')
assert args.wavfile != outfile

with open(outfile, 'wb') as f:
    f.write(bytes('C64-TAPE-RAW', 'utf8'))
    # version 0, no long pulse handling
    f.write(struct.pack('b', 0))
    # reserved
    f.write(bytes(3))
    # payload
    f.write(struct.pack('I', len(df.pl)))
    f.write(df.pl.to_numpy().tobytes())
