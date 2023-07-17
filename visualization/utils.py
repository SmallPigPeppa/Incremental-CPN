import os
import torch
import torch.nn as nn
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torchvision.models import resnet18
import warnings

seed_everything(7)
PATH_DATASETS = "/Users/lwz/torch_ds"
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)



def get_pretrained_encoder(ckpt_path, cifar=True):
    state = torch.load(ckpt_path,map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder.", "")] = state[k]
            warnings.warn(
                "You are using an older checkpoint. Use a new one as some issues might arrise."
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    encoder = resnet18()
    if cifar:
        encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        encoder.maxpool = nn.Identity()
    encoder.fc = nn.Identity()
    encoder.load_state_dict(state, strict=False)
    print(f"Loaded {ckpt_path}")
    return encoder

def get_cifar10_test_loader():
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )
    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=PATH_DATASETS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )
    cifar10_dm.prepare_data()
    cifar10_dm.setup()
    return cifar10_dm.test_dataloader()