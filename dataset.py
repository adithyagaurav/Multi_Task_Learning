import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class HydranetDataset(Dataset):
    def __init__(self, data_file, transform=None):
        super().__init__()
        img_paths = open(data_file).readlines()
        self.datalist = [x.strip("\n").split("\t") for x in img_paths]
        self.root_dir = "nyud"
        self.transform = transform
        self.mask_names = ("depth","segm")

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        abs_paths = [os.path.join(self.root_dir, rpath) for rpath in self.datalist[idx]]
        sample = {"image": np.array(Image.open(abs_paths[0]), dtype=np.float32),
                  "segm": np.array(Image.open(abs_paths[1]), dtype=np.float32),
                  "depth": np.array(Image.open(abs_paths[2]), dtype=np.float32),
                  "names":self.mask_names}
        if self.transform:
            sample = self.transform(sample)
            if "names" in sample:
                del sample["names"]
        return sample
