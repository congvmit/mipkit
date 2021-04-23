# Copyright (c) 2021 congvm
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataLoader(pl.LightningDataModule):
    def __init__(self, train_dataset,
                 val_dataset=None,
                 test_dataset=None,
                 batch_size=1,
                 num_workers=1,
                 pin_memory=False):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=True,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=self.num_workers,
                          pin_memory=True)
