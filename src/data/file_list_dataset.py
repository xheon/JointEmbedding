import os
from typing import List, Tuple

from torch.utils.data import Dataset

import data
import numpy as np

class FileListDataset(Dataset):
    def __init__(self, data_root: str, file_list: List[str], file_extension: str, transformation=None, ):
        super().__init__()

        self.data_root = data_root
        self.file_list = file_list
        self.transformation = transformation
        self.file_extension = file_extension

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int) -> Tuple[str, np.array]:
        name = self.file_list[index]
        element_path = os.path.join(self.data_root, name + self.file_extension)
        element = self.transformation(self._load(element_path)).tdf

        return name, element

    @staticmethod
    def _load(path):
        sample = data.load_raw_df(path)
        return sample

