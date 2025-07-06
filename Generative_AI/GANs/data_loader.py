import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import pydicom, torch
import os
import glob

class RSNATrainDataset(Dataset):
    def __init__(self, df, root, transform=None):
        self.df = df.drop_duplicates(subset='patientId')
        self.root = root
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        pid = self.df.iloc[idx].patientId
        dcm = pydicom.dcmread(f"{self.root}/{pid}.dcm")
        img = dcm.pixel_array.astype('float32')
        img = self.transform(img[..., None])
        return img


class RSNATestDataset(Dataset):
    def __init__(self, root, transform=None):
        # Grab all .dcm paths under root
        self.paths = sorted(glob.glob(os.path.join(root, "*.dcm")))
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        dcm  = pydicom.dcmread(path)
        img  = dcm.pixel_array.astype("float32")
        img  = self.transform(img[..., None])
        return img