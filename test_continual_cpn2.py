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
from test_utils import get_pretrained_dataset, get_pretrained_encoder, split_dataset
import argparse
import pytorch_lightning as pl


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
        logits=-1.*d
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

if __name__ == '__main__':
    num_classes = 100
    num_tasks = 5
    EPOCHS = 1000
    LAMBDA1= 0.2
    LAMBDA2 = 0.
    LR = 0.3
    BATCH_SIZE = 1024
    NUM_GPUS = [0,1]
    NUM_WORKERS = 1
    INCREMENTAL_N=10
    seed_everything(5)
    encoder = get_pretrained_encoder()
    train_dataset, test_dataset = get_pretrained_dataset(encoder=encoder)

    # split classes into tasks
    classes_order = torch.tensor(list(range(num_classes)))
    # classes_order = torch.randperm(num_classes)
    tasks_initial = classes_order[:int(num_classes / 2)].chunk(1)
    tasks_incremental = classes_order[int(num_classes / 2):num_classes].chunk(num_tasks)
    tasks = tasks_initial + tasks_incremental



    # mmodel = IncrementalPT(cn=50, dim_feature=2048, means=mds.get_means(classes=range(50)))




    mmodel = IncrementalPT(cn=50, dim_feature=2048, means=torch.rand([50,1024]))
    trainer = pl.Trainer(
        gpus=NUM_GPUS,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        accelerator='ddp',
        logger=TensorBoardLogger(f"./logs/", name=f"cifar_{int(50 / INCREMENTAL_N)}steps_0"),
        checkpoint_callback=False,
        precision=16,
    )
    train_dataset0 = split_dataset(
        train_dataset,
        tasks=tasks,
        task_idx=[0],
    )
    test_dataset0 = split_dataset(
        test_dataset,
        tasks=tasks,
        task_idx=list(range(0 + 1)),
    )
    train_loader = DataLoader(train_dataset0, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset0, batch_size=64, shuffle=True)
    trainer.fit(mmodel, train_loader, test_loader)


    for task_idx in range(1,num_tasks+1):
        EPOCHS = 300
        LAMBDA1 = 0.
        LAMBDA2 = 0.
        mmodel.upadate_w(incremental_cn=10, means=torch.rand([len(tasks[task_idx],1024)]))
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

        trainer = pl.Trainer(
            gpus=NUM_GPUS,
            max_epochs=EPOCHS,
            accumulate_grad_batches=1,
            sync_batchnorm=True,
            accelerator='ddp',
            logger=TensorBoardLogger(f"./logs/", name=f"cifar_{int(50 / INCREMENTAL_N)}steps_{num_tasks+1}"),
            checkpoint_callback=False,
            precision=16,

        )
        trainer.fit(mmodel, train_loader, test_loader)
