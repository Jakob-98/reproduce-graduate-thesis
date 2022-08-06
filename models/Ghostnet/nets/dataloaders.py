from importlib.resources import path
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from sklearn import metrics
import time
import copy
import os
import random
from matplotlib import pyplot as plt
import glob 
from pathlib import Path
import PIL
import math
import sys
import torchvision.transforms as transforms
from collections import Counter
from abc import abstractmethod
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Sequence, Union, Any, Callable
import torch.nn.functional as F


import yaml
import argparse
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin
import pytorch_lightning as pl
import torchvision.utils as vutils
from torch import optim, Tensor
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline



class DatasetLoader(Dataset):
    def __init__(self, transforms, imagepath, labelpath, histlbppath, nclass):
        self.imagepath = imagepath
        self.labelpath = labelpath
        self.histlbppath = histlbppath
        self.transforms = transforms
        self.nclass = nclass 
        self.imageids = sorted([os.path.basename(p).split('.jpg')[0] for p in glob.glob(imagepath + '/*.jpg')])

    
    # one hot encoder:
    def one_hot(self, label):
        one_hot = np.zeros(self.nclass)
        one_hot[label] = 1
        return one_hot

    def __getitem__(self, index):
        imageId = self.imageids[index]
        histlbp = np.load(Path(self.histlbppath) / (imageId + '.npy'), allow_pickle=False)
        with open(self.labelpath + imageId + '.txt') as f:
            target = f.readline()[0]
        target=self.one_hot(int(target))
        img = np.array(Image.open(os.path.join(self.imagepath, imageId + '.jpg')).convert('RGB'))
        img, target, histlbp = torch.Tensor(img).permute(2,0,1), torch.Tensor(target), torch.Tensor(histlbp)
        # return img, target
        img = torch.squeeze(self.transforms(img))
        return img, histlbp, target 


    def __len__(self):
        return len(self.imageids)

class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float() / 255.0

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = nn.Sequential(
            RandomHorizontalFlip(p=0.75),
            RandomChannelShuffle(p=0.75),
            RandomThinPlateSpline(p=0.75),
        )

        self.jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out

class DataModuleCustom(pl.LightningDataModule):
    def __init__(
        self,
        nclass,
        trainhistlbppath,
        trainlabelpath,
        trainimagepath,
        valhistlbppath,
        vallabelpath,
        valimagepath
    ):
        super().__init__()
        self.train_batch_size = 16
        self.num_workers = 1
        self.nclass = nclass
        self.trainhistlbppath = trainhistlbppath
        self.trainlabelpath = trainlabelpath
        self.trainimagepath = trainimagepath
        self.valhistlbppath = valhistlbppath
        self.vallabelpath = vallabelpath
        self.valimagepath = valimagepath
        self.transform = DataAugmentation()

    def setup(self, stage: Optional[str] = None) -> None:
        
        self.num_workers = 4
        self.train_dataset = DatasetLoader(
            transforms=self.transform, histlbppath=self.trainhistlbppath, imagepath=self.trainimagepath, labelpath=self.trainlabelpath, nclass=self.nclass
        )

        self.val_dataset= DatasetLoader(
            transforms=self.transform, histlbppath=self.valhistlbppath, imagepath=self.valimagepath, labelpath=self.vallabelpath, nclass=self.nclass
        )

        self.test_dataset= DatasetLoader(
            transforms=self.transform, histlbppath=self.valhistlbppath, imagepath=self.valimagepath, labelpath=self.vallabelpath, nclass=self.nclass
        )


        # self.val_dataset = ENA(
        # )

        # self.test_dataset = ENA(
        # )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            # num_workers=self.num_workers,
            # shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size = self.train_batch_size
        )
    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.train_batch_size
        )


    # def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.val_batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         pin_memory=self.pin_memory,
    #     )
    
    # def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     return DataLoader(
    #         self.test_dataset,
    #         batch_size=144,
    #         num_workers=self.num_workers,
    #         shuffle=True,
    #         pin_memory=self.pin_memory,
    #     )
     