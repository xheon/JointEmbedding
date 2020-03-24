import numpy as np

from data.sample import Sample


def truncation_normalization_transform(s: Sample) -> Sample:
    truncation = 3 * s.size
    s.tdf = np.abs(s.tdf)  # Omit sign
    s.tdf = np.clip(s.tdf, 0, truncation)  # Truncate
    s.tdf = s.tdf / truncation  # Normalize
    s.tdf = 1 - s.tdf  # flip

    return s


def to_flipped_occupancy_grid(s: Sample) -> Sample:
    s.tdf = np.abs(s.tdf) > s.size  # Omit sign
    s.tdf = s.tdf.astype(np.float32)
    s.tdf = s.tdf[:, :s.dimx, :s.dimy, :s.dimz]

    return s


def to_occupancy_grid(s: Sample) -> Sample:
    s.tdf /= s.size
    s.tdf = np.less_equal(np.abs(s.tdf), 1).astype(np.float32)
    s.tdf = s.tdf[:, :s.dimx, :s.dimy, :s.dimz]

    return s
