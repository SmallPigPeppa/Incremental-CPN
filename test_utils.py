import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import warnings
from torchvision.models import resnet18, resnet50
from torchvision import transforms
import torchvision
from torch.utils.data.dataset import Dataset, Subset
from typing import Callable, Optional, Tuple, Union, List
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch.nn.functional as F


def split_dataset(dataset: Dataset, task_idx: List[int], tasks: list = None):
    assert len(dataset.classes) == sum([len(t) for t in tasks])
    current_task = torch.cat(tuple(tasks[i] for i in task_idx))
    mask = [(c in current_task) for c in dataset.targets]
    indexes = torch.tensor(mask).nonzero()
    task_dataset = Subset(dataset, indexes)
    return task_dataset


def get_pretrained_encoder():
    ckpt_path = '/share/wenzhuoliu/code/solo-learn/trained_models/swav/yaaves5o/swav-imagenet32-yaaves5o-ep=999.ckpt'
    # ckpt_path = '/share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/s5fh5bvf/barlow_twins-imagenet32-s5fh5bvf-ep=999.ckpt'
    ckpt_path = '/share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238/byol-imagenet32-t3pmk238-ep=999.ckpt'
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
    print(f"Loaded {ckpt_path}")
    return encoder


def get_pretrained_dataset(encoder, train_dataset, test_dataset):
    IMGSIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    encoder.eval()
    encoder.to(device)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=8,
                             pin_memory=True)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    # encoder = nn.DataParallel(encoder)
    for x, y in tqdm(iter(train_loader), desc="pretrain on trainset"):
        x = x.to(device)
        z = encoder(x)
        x_train.append(z.cpu().detach().numpy())
        y_train.append(y.cpu().detach().numpy())
    for x, y in tqdm(iter(test_loader), desc="pretrain on testset"):
        x = x.to(device)
        z = encoder(x)
        x_test.append(z.cpu().detach().numpy())
        y_test.append(y.cpu().detach().numpy())

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.hstack(y_train)
    y_test = np.hstack(y_test)

    print("x_train.shape", x_train.shape, "y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape, "y_test.shape:", y_test.shape)
    # ds pretrained
    train_dataset_pretrained = TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset_pretrained = TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long))

    return train_dataset_pretrained, test_dataset_pretrained


class IncrementalPT(pl.LightningModule):
    def __init__(self, cn, dim_feature, means):
        super(IncrementalPT, self).__init__()
        self.dim_feature = dim_feature
        self.visable_cn = [0, cn - 1]
        self.w = nn.Parameter(torch.tensor(means))

    def forward(self, x):
        x = x.reshape(-1, 1, self.dim_feature)
        d = torch.pow(x - self.w, 2)
        d = torch.sum(d, dim=2)
        return d

    def incremental_forward(self, x):
        x = x.reshape(-1, 1, self.dim_feature)
        with torch.no_grad():
            c1 = self.w[0:self.visable_cn[0], :]
        c2 = self.w[self.visable_cn[0]:self.visable_cn[1] + 1, :]
        c = torch.cat((c1, c2), 0)
        d = torch.pow(x - c, 2)
        d = torch.sum(d, dim=2)
        return d

    def training_step(self, batch, batch_idx):
        x, targets = batch
        d = self.incremental_forward(x)
        logits = -1. * d
        # ce loss
        ce_loss = F.cross_entropy(logits, targets)
        # pl loss
        pl_loss = torch.index_select(d, dim=1, index=targets)
        pl_loss = torch.diagonal(pl_loss)
        pl_loss = torch.mean(pl_loss)
        # pl cos loss
        c = torch.index_select(self.w, dim=0, index=targets)
        cos_matrix = F.normalize(x, p=2, dim=1) * F.normalize(c, p=2, dim=1)
        pl_cosloss = -1. * torch.sum(cos_matrix)
        # all loss
        loss = ce_loss + pl_loss * LAMBDA1 + pl_cosloss * LAMBDA2
        self.log("c", ce_loss, prog_bar=True)
        self.log("p1", pl_loss, prog_bar=True)
        self.log("p2", pl_cosloss, prog_bar=True)
        # acc
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == targets) / targets.shape[0]
        self.log('acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=LR)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=opt, warmup_epochs=10, max_epochs=EPOCHS)
        return {'optimizer': opt, "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "metric_to_track",
        }}

    def evaluate(self, batch, stage=None):
        x, targets = batch
        d = self(x)
        logits = -1. * d
        # ce loss
        ce_loss = F.cross_entropy(logits, targets)
        # pl loss
        pl_loss = torch.index_select(d, dim=1, index=targets)
        pl_loss = torch.diagonal(pl_loss)
        pl_loss = torch.mean(pl_loss)
        # pl cos loss
        c = torch.index_select(self.w, dim=0, index=targets)
        cos_matrix = F.normalize(x, p=2, dim=1) * F.normalize(c, p=2, dim=1)
        pl_cosloss = -1. * torch.sum(cos_matrix)
        # acc
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == targets) / targets.shape[0]
        if stage:
            self.log('cv', ce_loss, prog_bar=True)
            self.log('p1v', pl_loss, prog_bar=True)
            self.log('p2v', pl_cosloss, prog_bar=True)
            self.log('vacc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def upadate_w(self, incremental_cn, means):
        incremental_w = nn.Parameter(torch.tensor(means))
        self.visable_cn = [self.visable_cn[1] + 1, self.visable_cn[1] + incremental_cn]
        self.w = nn.Parameter(torch.cat((self.w, incremental_w), dim=0), requires_grad=True)
