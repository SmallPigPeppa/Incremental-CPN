import torchvision

from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F
from torch import nn
import torch
from torchmetrics.functional import accuracy
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from test_utils import get_pretrained_dataset, get_pretrained_encoder, split_dataset,IncrementalPT
import argparse


def parse_args_linear() -> argparse.Namespace:
    """Parses feature extractor, dataset, pytorch lightning, linear eval specific and additional args.

    First adds an arg for the pretrained feature extractor, then adds dataset, pytorch lightning
    and linear eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_feature_extractor", type=str, required=True)
    parser.add_argument("--pretrain_method", type=str, default=None)

    # incremental
    parser.add_argument("--num_tasks", type=int, default=5)
    # parse args
    args = parser.parse_args()

    return args


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
        return loss + 0.03 * pl_loss

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

        return [optimizer]

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


if __name__ == '__main__':
    IMGSIZE = 32
    LR = 0.1
    GPUS = [0]
    BS0 = 128
    BS2 = 512
    num_classes = 100
    num_tasks = 5
    seed_everything(5)
    encoder = get_pretrained_encoder()
    train_dataset, test_dataset = get_pretrained_dataset(encoder=encoder)

    # split classes into tasks
    classes_order = torch.tensor(list(range(num_classes)))
    # classes_order = torch.randperm(num_classes)
    tasks_initial = classes_order[:int(num_classes / 2)].chunk(1)
    tasks_incremental = classes_order[int(num_classes / 2):num_classes].chunk(num_tasks)
    tasks = tasks_initial + tasks_incremental

    mmodel = IncrementalPT(cn=50, dim_feature=2048, means=mds.get_means(classes=range(50)))

    for task_idx in range(num_tasks + 1):
        train_dataset = split_dataset(
            train_dataset,
            tasks=tasks,
            task_idx=[task_idx],
        )
        test_dataset = split_dataset(
            test_dataset,
            tasks=tasks,
            task_idx=list(range(task_idx + 1)),
        )
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


        trainer = pl.Trainer(
            gpus=NUM_GPUS,
            max_epochs=EPOCHS,
            accumulate_grad_batches=1,
            sync_batchnorm=True,
            accelerator='ddp',
            logger=TensorBoardLogger(f"./logs/", name=f"cifar_{int(50 / INCREMENTAL_N)}steps_{i + 1}"),
            checkpoint_callback=False,
            precision=16,

        )
        trainer.fit(mmodel, train_loader, test_loader)
