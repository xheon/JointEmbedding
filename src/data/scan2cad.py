import json
import math
import os
import random
from collections import defaultdict
import numpy as np

from torch.utils.data import Dataset
from typing import List, Tuple, Any

import data


class Scan2Cad(Dataset):
    def __init__(self, dataset_file: str, scannet_root: str, shapenet_root: str,
                 name: str, splits=None, rotation=None, flip=False, jitter=False,
                 transformation=None, mask_scans=False, scan_rep="sdf", load_mask=False,
                 add_negatives=False) -> None:
        super().__init__()

        if splits is None:
            splits = ["train", "validation", "test"]

        self.rotation_augmentation = rotation
        self.flip_augmentation = flip
        self.jitter_augmentation = jitter

        self.mask_scans = mask_scans

        self.splits = splits
        self.scannet_root = scannet_root
        self.shapenet_root = shapenet_root
        self.scan_representation = scan_rep
        self.load_mask = load_mask
        self.dataset_file = dataset_file
        self.name = name

        if transformation is None:
            transformation = data.truncation_normalization_transform

        self.transformation = transformation

        self.models = {}
        self.pairs = self.load_from_json(self.dataset_file)

        self.has_negatives = add_negatives

        if self.has_negatives:
            self.negatives = self.add_negatives(self.pairs)

    @staticmethod
    def add_negatives(data: List[Tuple[str, str]]) -> List[str]:
        per_category = defaultdict(list)

        for scan, _ in data:
            category = scan.split("_")[4]
            per_category[category].append(scan)

        negatives = []
        for scan, _ in data:
            category = scan.split("_")[4]
            neg_categories = list(per_category.keys())
            neg_categories.remove(category)
            neg_category = np.random.choice(neg_categories)
            neg_cad = np.random.choice(per_category[neg_category])
            negatives.append(neg_cad)

        return negatives

    def regenerate_negatives(self) -> None:
        self.negatives = self.add_negatives(self.pairs)

    def load_from_json(self, file: str) -> List[Tuple[str, str]]:
        with open(file) as f:
            content = json.load(f)
            objects = content["scan2cad_objects"]

            pairs = []

            for k, v in objects.items():
                # if os.path.exists(os.path.join(self.scannet_root, k + ".mask")):
                if v in self.splits:
                    pair = (k, k)
                    pairs.append(pair)

        return pairs

    @staticmethod
    def get_shapenet_object(scannet_object_name: str) -> Any:
        parts = str.split(scannet_object_name, "_")

        if len(parts) == 6:
            return parts[4] + "/" + parts[5]
        else:
            return None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Any:
        objects = {}

        # Load scan sample
        scannet_object_name, shapenet_object_name = self.pairs[index]
        scannet_object_path = os.path.join(self.scannet_root, f"{scannet_object_name}.{self.scan_representation}")

        # Load scan mask
        if self.mask_scans or self.load_mask:
            scannet_mask_path = os.path.join(self.scannet_root, f"{scannet_object_name}.mask")
            objects["mask"], _ = self._load(scannet_mask_path)
        else:
            scannet_mask_path = None

        objects["scan"], _ = self._load(scannet_object_path, scannet_mask_path)

        # Load CAD sample
        shapenet_object_path = os.path.join(self.shapenet_root, f"{shapenet_object_name}.df")
        objects["cad"], _ = self._load(shapenet_object_path)

        # Load negative CAD sample
        if self.has_negatives:
            negative_name = self.negatives[index]
            negative_object_path = os.path.join(self.shapenet_root, f"{negative_name}.df")
            objects["negative"], _ = self._load(negative_object_path)
        else:
            negative_name = ""

        # Apply augmentations
        if self.rotation_augmentation == "interpolation":
            rotations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
            degree = random.choice(rotations)
            angle = degree * math.pi / 180
            objects = {k: data.rotation_augmentation_interpolation_v2(o, angle) for k, o in objects.items()}

        elif self.rotation_augmentation == "fixed":
            objects = {k: data.rotation_augmentation_fixed(o) for k, o in objects.items()}

        if self.flip_augmentation:
            objects = {k: data.flip_augmentation(o) for k, o in objects.items()}

        if self.jitter_augmentation:
            objects = {k: data.jitter_augmentation(o) for k, o in objects.items()}

        objects = {k: np.ascontiguousarray(o) for k, o in objects.items()}

        # Define final outputs
        scan_data = {"name": scannet_object_name, "content": objects["scan"]}

        if "mask" in objects:
            scan_data["mask"] = objects["mask"]

        cad_data = {"name": scannet_object_name, "content": objects["cad"]}

        if self.has_negatives:
            negative_data = {"name": negative_name, "content": objects["negative"]}
            return scan_data, cad_data, negative_data
        else:
            return scan_data, cad_data

    def _load(self, path, mask_path=None):
        model, info = self._load_df(path)
        if self.mask_scans and mask_path is not None:
            mask, mask_info = self._load_mask(mask_path)
            info.tdf = self.mask_object(model, mask)

        info = self.transformation(info)
        return info.tdf, info

    @staticmethod
    def mask_object(model, mask):
        masked = np.where(mask, model, np.NINF)

        return masked

    @staticmethod
    def _load_mask(filepath: str) -> Tuple[np.array, np.array]:
        mask = data.load_mask(filepath)

        return mask.tdf, mask

    @staticmethod
    def _load_df(filepath: str) -> Tuple[np.array, np.array]:
        if os.path.splitext(filepath)[1] == ".mask":
            sample = data.load_mask(filepath)
            sample.tdf = 1.0 - sample.tdf.astype(np.float32)
        else:
            sample = data.load_raw_df(filepath)
        patch = sample.tdf
        return patch, sample

    @staticmethod
    def _load_sdf(filepath: str) -> Tuple[np.array, np.array]:
        sample = data.load_sdf(filepath)
        patch = sample.tdf
        return patch, sample
