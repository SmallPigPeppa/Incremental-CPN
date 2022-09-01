import torch
import torchvision
import pytorch_lightning as pl
import argparse
import wandb
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils import get_pretrained_dataset, get_pretrained_encoder, split_dataset


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


class IncrementalPT(pl.LightningModule):
    def __init__(self, dim_feature, num_classes):
        super(IncrementalPT, self).__init__()
        self.dim_feature = dim_feature
        self.num_calsses = num_classes
        self.prototypes = nn.ParameterList(
            [nn.Parameter(torch.randn(1, self.dim_feature)) for i in range(num_classes)])

    def task_initial(self, current_tasks):
        no_grad_idx = [i for i in range(self.num_calsses) if i not in current_tasks]
        for i in no_grad_idx:
            self.prototypes[i].requires_grad = False

    def forward(self, x):
        x = x.reshape(-1, 1, self.features_dim)
        prototypes_list = [i for i in self.prototypes]
        d = torch.pow(x - torch.cat(prototypes_list), 2)
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
        self.log("c", ce_loss, on_epoch=True, sync_dist=True)
        self.log("p1", pl_loss, on_epoch=True, sync_dist=True)
        self.log("p2", pl_cosloss, on_epoch=True, sync_dist=True)
        # acc
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == targets) / targets.shape[0]
        self.log('acc', acc, on_epoch=True, sync_dist=True)
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
            self.log('cv', ce_loss, on_epoch=True, sync_dist=True)
            self.log('p1v', pl_loss, on_epoch=True, sync_dist=True)
            self.log('p2v', pl_cosloss, on_epoch=True, sync_dist=True)
            self.log('vacc', acc, on_epoch=True, sync_dist=True)

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
    EPOCHS = 300
    LAMBDA1 = 0.2
    LAMBDA2 = 0.
    LR = 0.3
    BATCH_SIZE = 1024
    NUM_GPUS = [0, 1]
    NUM_WORKERS = 1
    INCREMENTAL_N = 10
    IMGSIZE = 32
    seed_everything(5)
    encoder = get_pretrained_encoder()
    data_path = '/share/wenzhuoliu/torch_ds'

    # cifar100
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    # # imagenet
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    cifar_transforms = transforms.Compose(
        [transforms.Resize(IMGSIZE), transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                  transform=cifar_transforms,
                                                  download=True)
    test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                                 transform=cifar_transforms,
                                                 download=True)

    # split classes into tasks
    classes_order = torch.tensor(list(range(num_classes)))
    # classes_order = torch.randperm(num_classes)
    tasks_initial = classes_order[:int(num_classes / 2)].chunk(1)
    tasks_incremental = classes_order[int(num_classes / 2):num_classes].chunk(num_tasks)
    tasks = tasks_initial + tasks_incremental

    mmodel = IncrementalPT(dim_feature=2048, num_classes=num_classes)

    for task_idx in range(0, num_tasks + 1):
        EPOCHS = 300
        if task_idx == 0:
            LAMBDA1 = 0.3
        else:
            LAMBDA1 = 0.
        LAMBDA2 = 0.
        mmodel.task_initial(current_tasks=tasks[task_idx])
        wandb_logger = WandbLogger(
            name=f"task:{task_idx}",
            project="Incremental-CPN-v6",
            entity="pigpeppa",
            offline=False,
        )
        train_dataset_task = split_dataset(
            train_dataset,
            tasks=tasks,
            task_idx=[task_idx],
        )
        test_dataset_task = split_dataset(
            test_dataset,
            tasks=tasks,
            task_idx=list(range(task_idx + 1)),
        )
        train_dataset_task, test_dataset_task = get_pretrained_dataset(encoder=encoder,
                                                                       train_dataset=train_dataset_task,
                                                                       test_dataset=test_dataset_task)
        train_loader = DataLoader(train_dataset_task, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset_task, batch_size=64, shuffle=True)

        trainer = pl.Trainer(
            gpus=NUM_GPUS,
            max_epochs=EPOCHS,
            accumulate_grad_batches=1,
            sync_batchnorm=True,
            accelerator='ddp',
            logger=wandb_logger,
            checkpoint_callback=False,
            precision=16,

        )
        trainer.fit(mmodel, train_loader, test_loader)
        wandb.finish()
