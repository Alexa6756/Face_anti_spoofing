import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os

class FaceAntiSpoofDataset(Dataset):
    def __init__(self, csv_file, root_dir, split='train', transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        color_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        label = int(self.data.iloc[idx, 3])

        img = cv2.imread(color_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {color_path}")
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.0  

        return torch.tensor(img, dtype=torch.float32), label
