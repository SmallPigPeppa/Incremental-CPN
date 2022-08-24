import pl_bolts
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F
from torch import nn
import torch
from torchmetrics.functional import accuracy
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from torchvision.models import resnet18, resnet50
import warnings
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from pytorch_lightning.loggers import WandbLogger


class MLP(LightningModule):
    def __init__(self, dim_in=2048, dim_out=100):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.model = nn.Linear(dim_in, dim_out)

        # ckpt_path = '/share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/1ehqqmug/barlow_twins-imagenet32-1ehqqmug-ep=437.ckpt'
        # ckpt_path = '/share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/1ehqqmug/barlow_twins-imagenet32-1ehqqmug-ep=999.ckpt'
        ckpt_path = '/share/wenzhuoliu/code/solo-learn/trained_models/simclr/2mv95572/simclr-imagenet32-2mv95572-ep=999.ckpt'

        state = torch.load(ckpt_path)["state_dict"]
        for k in list(state.keys()):
            if "encoder" in k:
                state[k.replace("encoder", "backbone")] = state[k]
                warnings.warn(
                    "You are using an older checkpoint. Use a new one as some issues might arrise."
                )
            if "backbone" in k:
                state[k.replace("backbone.", "")] = state[k]
            del state[k]

        encoder = resnet50()
        encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        encoder.maxpool = nn.Identity()
        encoder.fc = nn.Identity()
        encoder.load_state_dict(state, strict=False)

        self.encoder = encoder

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=LR,
            momentum=0.9,
            weight_decay=0.,
        )
        self.scheduler = "step"
        self.lr_decay_steps = [30, 60]
        # select scheduler
        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * self.num_training_steps
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.max_epochs * self.num_training_steps
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

        # return [optimizer]


import os

import os.path

if __name__ == '__main__':
    IMGSIZE = 32
    LR = 0.1
    GPUS = [0]
    BS0 = 128
    BS2 = 512
    data_path = '/share/wenzhuoliu/torch_ds'
    # cifar100
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    # # imagenet
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    cifar_transforms = transforms.Compose(
        [transforms.Resize(IMGSIZE), transforms.ToTensor(), transforms.Normalize(mean, std)])
    # transforms.CenterCrop(size=96)

    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                  transform=cifar_pipeline["T_train"],
                                                  download=True)
    test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                                 transform=cifar_pipeline["T_val"],
                                                 download=True)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=8,
                             pin_memory=True)

    model = MLP()
    wandb_logger = WandbLogger(
        name="joint-train-cifar-simclr",
        project="Incremental-CPN",
        entity="pigpeppa",
        offline=False,
        resume=None,
        id=None,
    )
    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=100,
        gpus=GPUS,
        logger=wandb_logger,
        checkpoint_callback=False,
        precision=16
    )

    trainer.fit(model, train_loader, test_loader)
