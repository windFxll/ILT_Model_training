import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ILTDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir:
            D:/S-Litho_python/ILT_Model_training/mental_layer_input/png
        """

        self.root_dir = os.path.abspath(root_dir)

        self.image_paths = sorted(
            glob.glob(os.path.join(self.root_dir, "*.png"))
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No png files found in: {self.root_dir}"
            )

        print(f"[Dataset] Found {len(self.image_paths)} images")

        # debug
        print("[Dataset] First sample:", self.image_paths[0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        img = cv2.imread(img_path, 0)

        if img is None:
            raise RuntimeError(
                f"Failed to read image: {img_path}"
            )

        # 转 float32
        img = img.astype(np.float32)

        # binary normalize
        img = img / 255.0

        # [H,W] -> [1,H,W]
        img = torch.tensor(img).unsqueeze(0)

        return img