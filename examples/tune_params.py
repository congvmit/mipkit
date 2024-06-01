"""
The MIT License (MIT)
Copyright (c) 2021 Cong Vo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

Provided license texts might have their own copyrights and restrictions

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import gc
import math
import os

import pytorch_lightning as pl
import torch
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from mipkit.dl.tune import start_tuning
from mipkit.utils import generate_datetime

gc.collect()
torch.cuda.empty_cache()

# ===============================================================================
# Modeling
# ===============================================================================


class LightningMNISTClassifier(pl.LightningModule):
    """
    This has been adapted from
    https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
    """

    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()

        self.data_dir = data_dir or os.getcwd()
        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.opt = config["opt"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = torch.relu(x)

        x = self.layer_2(x)
        x = torch.relu(x)

        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss, on_epoch=True)
        self.log("ptl/train_accuracy", accuracy, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/val_loss", loss, on_epoch=True)
        self.log("ptl/val_accuracy", accuracy, on_epoch=True)

    @staticmethod
    def download_data(data_dir):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return MNIST(data_dir, train=True, download=False, transform=transform)

    def prepare_data(self):
        mnist_train = self.download_data(self.data_dir)

        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=int(self.batch_size), num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=int(self.batch_size), num_workers=4)

    def configure_optimizers(self):
        if self.opt == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.opt == "sgd":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train_mnist_tune(tuning_config, data_dir=None, num_epochs=10, num_gpus=1):
    # Only Training
    model = LightningMNISTClassifier(tuning_config, data_dir)

    # ===============================================================================
    # Callback
    # ===============================================================================
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    early_stop_cb = EarlyStopping(monitor="ptl/val_loss", patience=5, verbose=True, mode="min")

    ckpt_cb = ModelCheckpoint(
        tune.get_trial_dir() + "/checkpoints",
        save_top_k=5,
        verbose=True,
        monitor="ptl/val_loss",
        mode="min",
        save_last=True,
        filename="model_{epoch:03d}-{step}",
    )

    tune_rp_cb = TuneReportCallback(
        {"val_loss": "ptl/val_loss", "val_accuracy": "ptl/val_accuracy"},
        on="validation_end",
    )

    # ===============================================================================
    # Trainer
    # ===============================================================================
    trainer = pl.Trainer(
        progress_bar_refresh_rate=0,  # 0 means no print progress
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        callbacks=[ckpt_cb, tune_rp_cb, early_stop_cb],
    )

    trainer.logger._default_hp_metric = False  # hp_metrc must be False
    trainer.fit(model)


if __name__ == "__main__":
    # ===============================================================================
    # Start Process
    # ===============================================================================

    train_config = {"data_dir": "<data_path>", "num_epochs": 40, "num_gpus": 1}

    tuning_config = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "opt": tune.choice(["adam", "sgd"]),
    }

    log_dir = "experiments"
    experiment_name = "tune_mnist_asha_" + generate_datetime()

    metric_columns = ["val_loss", "val_accuracy", "training_iteration"]

    start_tuning(
        tuning_config=tuning_config,
        train_config=train_config,
        training_func=train_mnist_tune,
        report_metric_columns=metric_columns,
        monitor_metric="val_loss",
        monitor_mode="min",
        log_dir=log_dir,
        experiment_name=experiment_name,
        num_epochs=20,
        num_samples=2,
        cpus_per_trial=2,  # Optimize with num_workers args
        gpus_per_trial=0.2,
    )
