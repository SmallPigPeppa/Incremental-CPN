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


class CPN(LightningModule):
    def __init__(self, features_dim=2048, num_classes=100):
        super().__init__()
        self.features_dim = features_dim
        self.num_calsses = num_classes
        self.prototypes = nn.ParameterList(
            [nn.Parameter(torch.randn(1, self.features_dim)) for i in range(self.num_calsses)])

    def forward(self, x):
        x = x.reshape(-1, 1, self.features_dim)
        prototypes_list = [i for i in self.prototypes]
        d = torch.pow(x - torch.cat(prototypes_list), 2)
        d = torch.sum(d, dim=2)
        # logits = -1. * d
        # return logits
        return d

    def training_step(self, batch, batch_idx):
        x, y = batch
        d = self(x)
        loss = F.cross_entropy(-1.0 * d, y)

        pl_loss = torch.index_select(d, dim=1, index=y)
        pl_loss = torch.diagonal(pl_loss)
        pl_loss = torch.mean(pl_loss)
        self.log("train_loss", loss)
        return loss + 0.2 * pl_loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = -1.0 * self(x)
        # loss = F.nll_loss(logits, y)
        loss = F.cross_entropy(logits, y)
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


class PrototypeClassifier(nn.Module):
    def __init__(self, features_dim, num_classes):
        super().__init__()
        self.features_dim = features_dim
        self.num_calsses = num_classes
        self.prototypes = nn.ParameterList(
            [nn.Parameter(torch.randn(1, self.features_dim)) for i in range(self.num_calsses)])

    def forward(self, x):
        x = x.reshape(-1, 1, self.features_dim)
        prototypes_list = [i for i in self.prototypes]
        squared_Euclidean_distance = torch.pow(x - torch.cat(prototypes_list), 2)
        squared_Euclidean_distance = torch.sum(squared_Euclidean_distance, dim=2)
        logits = -1. * squared_Euclidean_distance
        return logits

    def incremental_initial(self, means=None, current_tasks=list(range(10))):
        if means is not None:
            for i in current_tasks:
                nn.init.constant_(self.prototypes[i].data, means[i])
        no_grad_idx = [i for i in range(self.num_calsses) if i not in current_tasks]
        for i in no_grad_idx:
            self.prototypes[i].requires_grad = False


import os

import os.path

if __name__ == '__main__':
    IMGSIZE = 32
    LR = 0.1
    GPUS = [0]
    BS0 = 128
    BS2 = 512
    # ckpt_dir='/mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/simclr/2mv95572'
    # ckpt_dir='/share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238'
    # ckpt_dir='/share/wenzhuoliu/code/solo-learn/trained_models/mocov2plus/1kguyx5e'
    ckpt_dir = '/share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/1ehqqmug'

    # ckpt_dir='/share/wenzhuoliu/code/'
    data_path = '/share/wenzhuoliu/torch_ds'

    # for filename in os.listdir(ckpt_dir):
    #     basename, ext = os.path.splitext(filename)
    #     if ext == '.ckpt':
    #         ckpt_path = os.path.join(ckpt_dir, filename)
    #         print(f'load ckpt from {ckpt_path}')

    ckpt_path = '/share/wenzhuoliu/code/solo-learn/trained_models/swav/yaaves5o/swav-imagenet32-yaaves5o-ep=999.ckpt'
    # ckpt_path = '/share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/s5fh5bvf/barlow_twins-imagenet32-s5fh5bvf-ep=999.ckpt'

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
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    encoder.eval()
    encoder.to(device)
    # cifar100
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    # # imagenet
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    cifar_transforms = transforms.Compose(
        [transforms.Resize(IMGSIZE), transforms.ToTensor(), transforms.Normalize(mean, std)])
    # transforms.CenterCrop(size=96)
    train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                  transform=cifar_transforms,
                                                  download=True)
    test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                                 transform=cifar_transforms,
                                                 download=True)
    # train_dataset = torchvision.datasets.CIFAR100(root='~/torch_ds', split='train',
    #                                                            transform=stl_transform,
    #                                                            download=True)
    # test_dataset = torchvision.datasets.CIFAR100(root='~/torch_ds', split='test',
    #                                                            transform=stl_transform,
    #                                                            download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=8,
                             pin_memory=True)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    encoder = nn.DataParallel(encoder)
    for x, y in tqdm(iter(train_loader)):
        x = x.to(device)
        z = encoder(x)
        x_train.append(z.cpu().detach().numpy())
        y_train.append(y.cpu().detach().numpy())
    for x, y in tqdm(iter(test_loader)):
        x = x.to(device)
        z = encoder(x)
        x_test.append(z.cpu().detach().numpy())
        y_test.append(y.cpu().detach().numpy())

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.hstack(y_train)
    y_test = np.hstack(y_test)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    # ds pretrained
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = CPN()
    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=100,
        gpus=GPUS,
        logger=TensorBoardLogger(f"./logs/", name=f"linear-eval-cifar"),
        checkpoint_callback=False
    )

    trainer.fit(model, train_loader, test_loader)

    '''
    x, y = next(iter(train_loader))
    print(x)
    print(y)
    out = simclr_resnet50(x)
    print(out)
    '''
