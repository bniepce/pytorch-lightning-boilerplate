import os
import pytorch_lightning as pl
from src.datasets.image_detection import ImageDetectionDataset
from src.utils.detection import collate_fn, get_train_transform
from torch.utils.data import random_split,  DataLoader
from typing import Optional

class ImageDetectionDataModule(pl.LightningDataModule):
    """
    ImageDetectionDataModule used for image detection
    """
    def __init__(self, classes, batch_size = 4, resize = 512, num_workers = 0,  data_dir = None, normalize = False):
        super().__init__()
        self.data_dir = data_dir if data_dir is not None else os.path.join(os.getcwd(), 'data')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.classes = classes
        self.resize = resize
        self.X_train, self.X_val = [..., ...]

    def prepare_data(self):
        """
        Download and save data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Setup the pre-defined Pytorch Datasets
        """
        if stage == "fit" or stage is None:
            train_set_full = ImageDetectionDataset(os.path.join(self.data_dir, 'train'), self.resize, self.resize, self.classes, get_train_transform())
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.X_train, self.X_val = random_split(train_set_full, [train_set_size, valid_set_size])
            
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.X_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers, collate_fn=collate_fn)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.X_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers, collate_fn=collate_fn)