import os
import struct

import numpy as np
from typing import Tuple, AnyStr


class Sample:
    filename: str = ""
    dimx: int = 0
    dimy: int = 0
    dimz: int = 0
    size: float = 0.0
    matrix: np.array = None
    tdf: np.array = None
    sign: np.array = None


def save_tdf(filename: str, tdf: np.array, dimx: int, dimy: int, dimz: int, voxel_size: float, matrix: np.array) -> None:
    with open(filename, 'wb') as f:
        f.write(struct.pack('I', dimx))
        f.write(struct.pack('I', dimy))
        f.write(struct.pack('I', dimz))
        f.write(struct.pack('f', voxel_size))
        f.write(struct.pack("={}f".format(16), *matrix.flatten("F")))

        num_elements = dimx * dimy * dimz
        f.write(struct.pack("={}f".format(num_elements), *tdf.flatten("F")))


def load_sample_info(filename: str) -> Tuple[Sample, AnyStr, int]:
    assert os.path.isfile(filename), "File not found: %s" % filename
    content = open(filename, "rb").read()

    sample = Sample()
    sample.filename = filename
    sample.dimx = struct.unpack('I', content[0:4])[0]
    sample.dimy = struct.unpack('I', content[4:8])[0]
    sample.dimz = struct.unpack('I', content[8:12])[0]
    sample.size = struct.unpack('f', content[12:16])[0]

    matrix_size = int(16 * 4)
    sample.matrix = struct.unpack('f' * 16, content[16:16 + matrix_size])
    sample.matrix = np.asarray(sample.matrix, dtype=np.float32).reshape([4, 4])

    start_index = 16 + matrix_size

    return sample, content, start_index


def load_raw_df(filename: str) -> Sample:
    s, raw, start_index = load_sample_info(filename)
    n_elements = s.dimx * s.dimy * s.dimz

    # Load distance values
    s.tdf = struct.unpack('f' * n_elements, raw[start_index:start_index + n_elements * 4])
    s.tdf = np.asarray(s.tdf, dtype=np.float32).reshape([1, s.dimz, s.dimy, s.dimx])
    s.tdf = s.tdf.transpose((0, 3, 2, 1))

    return s


def load_mask(filename: str) -> Sample:
    s, raw, start_index = load_sample_info(filename)
    n_elements = s.dimx * s.dimy * s.dimz

    # Load binary values
    s.tdf = struct.unpack('B' * n_elements, raw[start_index:start_index + n_elements])
    s.tdf = np.asarray(s.tdf, dtype=np.dtype("?")).reshape([1, s.dimz, s.dimy, s.dimx])
    s.tdf = s.tdf.transpose((0, 3, 2, 1))

    return s


# Encode sign in a separate channel
def load_sdf(filename: str) -> Sample:
    s = load_raw_df(filename)

    truncation = 3 * s.size
    s.sign = np.sign(s.tdf)  # Encode sign in separate channel
    s.sign[s.sign >= 0] = 1
    s.sign[s.sign < 0] = 0
    s.tdf = np.abs(s.tdf)  # Omit sign
    s.tdf = np.clip(s.tdf, 0, truncation)  # Truncate
    s.tdf = s.tdf / truncation  # Normalize
    s.tdf = 1 - s.tdf  # flip
    s.tdf = s.tdf[:, :s.dimx, :s.dimy, :s.dimz]
    s.tdf = np.concatenate((s.tdf, s.sign), axis=0)

    return s
