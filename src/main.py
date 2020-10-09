#!/usr/bin/env python
'''
Outer ear simulator

Author: Michal Sudwoj <msudwoj@student.ethz.ch>
Version: 1.0.0
Data: 2019-09-09
'''

from typing import Tuple
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as ss
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pysofaconventions import SOFAFile

def main() -> None:
    args = arg_parser().parse_args()
    data, f_s = read(args.input_file)

    if args.head:
        data = head(data, args.sofa, args.azimuth, args.elevation)
    if args.canal:
        data = canal(data, f_s, args.l, args.d)
    if args.middle:
        data = middle(data)

    wav.write(args.output_file, f_s, data)

def head(data : np.ndarray, sofa : SOFAFile, azimuth : float, elevation : float):
    '''
    Apply effects of the head (HRTF)
    '''
    from scipy.spatial import KDTree

    s = get_sofa(sofa)
    pos = s.getVariableValue('SourcePosition')
    # find closest position to requested azimuth and elevation
    # TODO: consider normalizing position units to eg. degrees
    index = KDTree(pos).query([azimuth, elevation, 1])[1]
    hrir = s.getDataIR()[index, :, :]

    data = data.T
    left = ss.fftconvolve(data, hrir[0])
    right = ss.fftconvolve(data, hrir[1])
    output = np.asarray([left, right]).swapaxes(-1, 0)

    return output

def canal(input : np.ndarray, f_s: int, l : float, d : float):
    '''
    Apply effects of the ear canal

    Modeled as a bandpass filter, as in 'Matlab Auditory Periphery (MAP)'
    '''
    assert f_s > 0
    assert l >= 0
    assert d >= 0

    v = 343
    gain = 10
    order = 1
    f_nyq = f_s / 2

    for n in [1, 3, 5]:
        # 'Stopped pipe' resonator; resonating frequency
        f_r = (n * v) / (4 * l / 1000 + 0.4 * d / 1000)
        # bandpass cut offsets somewhat chosen s.t. for the first mode, they coincide with the parameters from MAP
        lowcut = f_r - 1500 # Hz
        highcut = f_r + 500 # Hz

        low = lowcut / f_nyq
        high = highcut / f_nyq
        b, a = ss.butter(order, [low, high], btype = 'band')
        input += gain * ss.lfilter(b, a, input)

    return input

def middle(input):
    '''
    Apply the effects of the middle ear

    Modelled soley as impedence mismatch and lever
    '''
    z_air = 414 # kg m^-2 s^-1
    z_water = 1.48e6 # kg m^-2 s^-1
    A_eardrum = 60 # mm^2
    A_oval = 3.2 # mm^2
    lever_malleus = 1.3

    reflected = ((z_air - z_water) / (z_air + z_water)) ** 2
    transmitted = 1 - reflected

    return input * transmitted * (A_eardrum / A_oval) * lever_malleus

def arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        formatter_class = ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--head',
        help = 'Consider head effects',
        dest = 'head',
        action = 'store_true'
    )
    parser.add_argument(
        '--no-head',
        dest = 'head',
        action = 'store_false'
    )
    parser.set_defaults(head = True)
    parser.add_argument(
        '--canal',
        help = 'Consider ear canal effects',
        dest = 'canal',
        action = 'store_true'
    )
    parser.add_argument(
        '--no-canal',
        dest = 'canal',
        action = 'store_false'
    )
    parser.set_defaults(canal = True)
    parser.add_argument(
        '--middle',
        help = 'Consider middle ear effects',
        dest = 'middle',
        action = 'store_true'
    )
    parser.add_argument(
        '--no-middle',
        dest = 'middle',
        action = 'store_false'
    )
    parser.set_defaults(middle = True)

    parser.add_argument(
        '--sofa',
        help = 'HTRF Sofa file',
        default = 'http://sofacoustics.org/data/database/cipic/subject_003.sofa'
    )
    parser.add_argument(
        '-a', '--azimuth',
        help = 'Azimuth of source in SOFA file units',
        default = 0,
        type = float
    )
    parser.add_argument(
        '-e', '--elevation',
        help = 'Elevation of source in SOFA file units',
        default = 0,
        type = float
    )

    parser.add_argument(
        '-l',
        help = 'Ear canal length in mm',
        default = 22,
        type = float
    )
    parser.add_argument(
        '-d',
        help = 'Ear canal diameter in mm',
        default = 7,
        type = float
    )

    parser.add_argument(
        'input_file',
        help = 'Input file'
    )
    parser.add_argument(
        'output_file',
        help = 'Output file'
    )

    return parser

def read(filename : str) -> Tuple[np.ndarray, float]:
    '''
    Read WAV file and normalize to float array
    '''
    f_s, data = wav.read(filename)
    if data.dtype == 'uint8':
        data = data / 255 - 0.5
    elif data.dtype == 'int16':
        data = data / 32767
    elif data.dtype == 'int32':
        data = data / 2147483647
    elif data.dtype == 'float32':
        data = 1.0 * data
    else:
        eprint(f'Input error: data.dtype = {data.dtype}')
        exit(1)

    if data.ndim == 1:
        # mono
        pass
    elif data.ndim == 2:
        data = data[:, 0]
    else:
        eprint(f'Input error: data.ndim = {data.ndim}')
        exit(1)

    return data, f_s

def get_sofa(url : str) -> SOFAFile:
    import requests
    from tempfile import NamedTemporaryFile

    if url.startswith(('http://', 'https://')):
        r = requests.get(url)
        r.raise_for_status()

        with NamedTemporaryFile() as f:
            f.write(r.content)
            return SOFAFile(f.name, 'r')
    elif url.startswith('file://'):
        url = url[7:]
    return SOFAFile(url, 'r')

def eprint(*args, **kwargs):
    from sys import stderr
    print(*args, file = stderr, **kwargs)


if __name__ == "__main__":
    main()
