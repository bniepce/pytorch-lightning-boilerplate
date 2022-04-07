import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from typing import Optional
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, num_workers, normalize = False, data_dir = None):
        super().__init__()
        self.data_dir = data_dir if data_dir is not None else os.path.join(os.getcwd(), 'data')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = 5000
        self.normalize = normalize
        self.X_train, self.X_val, self.X_test = [..., ..., ...]

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train = True, download = True)
        MNIST(self.data_dir, train = False, download = True)
    
    def setup(self, stage: Optional[str] = None):
        extra = dict(transform=self.default_transforms) if self.default_transforms else {}
        dataset = MNIST(self.data_dir, train=True, download=False, **extra)
        train_length = len(dataset)
        self.X_train, self.X_val = random_split(dataset, [train_length - self.val_split, self.val_split])
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.X_train, batch_size = self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.X_val, batch_size = self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.X_test, batch_size = self.batch_size, num_workers=self.num_workers)

    @property
    def default_transforms(self):
        if not _TORCHVISION_AVAILABLE:
            return None
        if self.normalize:
            mnist_transforms = transform_lib.Compose(
                [transform_lib.ToTensor(), transform_lib.Normalize(mean=(0.5,), std=(0.5,))]
            )
        else:
            mnist_transforms = transform_lib.ToTensor()

        return mnist_transforms