# Copyright 2022 Josh Bailey (josh@vandervecken.com)

"""Command-line entry point for wav2tapsp."""

import argparse

from wav2tapsp.convert import DEFAULT_CPUFREQ, read_wav_to_tap


def main(argv=None):
    parser = argparse.ArgumentParser(description='Convert WAV file into a C64 .tap file')
    parser.add_argument('wavfile', help='input WAV file')
    parser.add_argument('--cpufreq', default=int(DEFAULT_CPUFREQ), type=int, help='CPU frequency in Hz')
    args = parser.parse_args(argv)

    tap = read_wav_to_tap(args.wavfile, args.cpufreq)

    outfile = args.wavfile.replace('.wav', '.tap')
    assert args.wavfile != outfile

    with open(outfile, 'wb') as f:
        f.write(tap)


if __name__ == '__main__':
    main()
