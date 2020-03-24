import math
import random

import numpy as np
import scipy
import torch
from typing import List


def random_rotation() -> int:
    return np.random.randint(0, 4)


def random_degree() -> int:
    return np.random.randint(0, 360)


def random_angle() -> float:
    return np.random.random() * 2 * math.pi


def random_jitter(max_jitter: int = 4) -> List[int]:
    jitter = [np.random.randint(-max_jitter, max_jitter) for _ in range(3)]
    return jitter


def get_rotations_y(angles: np.array) -> np.array:
    cos_angle = np.cos(angles)
    sin_angle = np.sin(angles)
    rots = np.eye(3)[np.newaxis, :]
    rots = np.tile(rots, [angles.shape[0], 1, 1])
    rots[:, 0, 0] = cos_angle
    rots[:, 0, 2] = sin_angle
    rots[:, 2, 0] = -sin_angle
    rots[:, 2, 2] = cos_angle
    return rots.astype(np.float32)


def rotation_augmentation_fixed(grid: np.array, num_rotations=None) -> np.array:
    if num_rotations is None:
        angle = np.random.randint(0, 4)
    else:
        angle = num_rotations

    grid = np.rot90(grid, k=angle, axes=(1, 3))
    return grid


def rotate_grid(grid: np.array, num_rotations: int) -> np.array:
    patch = np.rot90(grid, k=num_rotations, axes=(1, 3))
    return patch


def rotation_augmentation_interpolation_v2(grid: np.array, rotation=None) -> np.array:
    if rotation is None:
        rotation = random_angle()

    scans = torch.from_numpy(np.expand_dims(grid, axis=0))
    num = scans.shape[0]
    rots = np.asarray([rotation])
    rotations_y = torch.from_numpy(get_rotations_y(rots))
    max_size = np.array(scans.shape[2:], dtype=np.int32)
    center = (max_size - 1).astype(np.float32) * 0.5
    center = np.tile(center.reshape(3, 1), [1, max_size[0] * max_size[1] * max_size[2]])
    grid_coords = np.array(
        np.unravel_index(np.arange(max_size[0] * max_size[1] * max_size[2]), [max_size[0], max_size[1], max_size[2]]),
        dtype=np.float32) - center
    grid_coords = np.tile(grid_coords[np.newaxis, :], [num, 1, 1])
    grid_coords = torch.from_numpy(grid_coords)
    center = torch.from_numpy(center).unsqueeze(0).repeat(scans.shape[0], 1, 1)
    grid_coords = torch.bmm(rotations_y, grid_coords) + center
    grid_coords = torch.clamp(grid_coords, 0, max_size[0] - 1).long()
    grid_coords = grid_coords[:, 0] * max_size[1] * max_size[2] + grid_coords[:, 1] * max_size[2] + grid_coords[:, 2]
    mult = torch.arange(num).view(-1, 1) * max_size[0] * max_size[1] * max_size[2]
    grid_coords = grid_coords + mult
    grid_coords = grid_coords.long()
    scan_rots = scans.permute(0, 2, 3, 4, 1).contiguous().view(-1, 1)[grid_coords]
    scan_rots = scan_rots.view(scans.shape[0], scans.shape[2], scans.shape[3], scans.shape[4], scans.shape[1]).permute(
        0, 4, 1, 2, 3)
    scan_rots = scan_rots.numpy()
    return scan_rots[0]


def rotation_augmentation_interpolation(grid: np.array, rotation=None) -> np.array:
    if rotation is None:
        angle = random_degree()
    else:
        angle = rotation
    grid = scipy.ndimage.rotate(grid, angle, (1, 3), False, prefilter=True, order=3, cval=0, mode="nearest")
    return grid


def flip_augmentation(grid: np.array, flip=None) -> np.array:
    if flip is None:
        chance = random.random() < 0.5
    else:
        chance = flip

    if chance:
        grid = np.flip(grid, (1, 3))

    return grid


def jitter_augmentation(grid: np.array, jitter=None) -> np.array:
    if jitter is None:
        jitter = random_jitter()

    start = [max(0, j) for j in jitter]
    end = [max(0, -j) for j in jitter]
    pad = np.pad(grid, ((0, 0),
                        (start[0], end[0]),
                        (start[1], end[1]),
                        (start[2], end[2])), "constant", constant_values=(0, 0))

    offset_start = [max(0, -j) for j in jitter]
    offset_end = [None if max(0, j) == 0 else -j for j in jitter]
    grid = pad[:, offset_start[0]:offset_end[0], offset_start[1]:offset_end[1], offset_start[2]:offset_end[2]]

    return grid
