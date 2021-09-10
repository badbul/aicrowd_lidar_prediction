import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl
from .transforms import default_transforms



class PointDataset(Dataset):
    def __init__(self, data, transforms=default_transforms()):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = self.data[idx][0]
        if self.transforms:
            pointcloud = self.transforms(pointcloud)
        label = self.data[idx][1]
        label = torch.from_numpy(np.array([label]))
        return {'pointcloud': pointcloud.float(),
                'label': label.float()}


class PointTestDataset(Dataset):
    def __init__(self, data, transforms=default_transforms()):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = self.data[idx]
        if self.transforms:
            pointcloud = self.transforms(pointcloud)

        return {'pointcloud': pointcloud.float(),
                'index': idx
                }


class PointDataModule(pl.LightningDataModule):
    """Datamodule for faults segmentation.
    Config:
        - batch_size: How many samples per batch to load.
    """

    def __init__(self, data_path, valid_size=0.1,  config=None):
        super().__init__()
        self.config = config  # prepare_config(self, config)
        self.data_path = data_path
        self.num_workers = self.config['num_workers']
        self.valid_size = valid_size
        self.data = None
        self.train_data = None
        self.valid_data = None

    def _load_data(self, stage):

        if stage == 'fit':
            self.data = np.load(self.data_path + "train.npz", allow_pickle=True)
            self.data = self.data['train']
            indices = np.arange(self.data.shape[0])
            np.random.seed(42)
            np.random.shuffle(indices)
            self.train_data = self.data[indices[:-int(self.valid_size * len(indices))]]
            self.valid_data = self.data[indices[-int(self.valid_size * len(indices)):]]

        else:
            self.data = np.load(self.data_path + "test.npz", allow_pickle=True)
            self.data = self.data['test']

    def setup(self, stage=None):
        if stage == 'test':
            self._load_data(stage)
            self.test_dataset = PointTestDataset(
                self.data,
                default_transforms()
            )

        if stage == 'fit' or stage is None:
            self._load_data(stage)

            self.train_dataset = PointDataset(
                self.train_data,
                default_transforms()
            )

            self.valid_dataset = PointDataset(
                self.valid_data,
                default_transforms()
            )


    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.num_workers,
            shuffle=False
        )


    # Loaders for auto labeling
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.num_workers,
            shuffle=False
        )

