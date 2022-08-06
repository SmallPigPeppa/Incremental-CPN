import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import multiprocessing
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
import torchvision
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
from metrics import accuracy_at_k, weighted_mean
import warnings
import numpy as np
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from utils.lars import LARSWrapper
from functools import partial
from utils.misc import compute_dataset_size

def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class LinearModel(pl.LightningModule):
    def __init__(self,
                 encoder: str,
                 num_classes: int,
                 max_epochs: int,
                 batch_size: int,
                 optimizer: str,
                 lr: float,
                 weight_decay: float,
                 extra_optimizer_args: Dict,
                 scheduler: str,
                 tasks: list,
                 num_tasks: int,
                 min_lr: float = 0.0,
                 warmup_start_lr: float = 0.00003,
                 warmup_epochs: float = 10,
                 scheduler_interval: str = "step",
                 lr_decay_steps: Sequence = None,
                 no_channel_last: bool = False,
                 **kwargs, ):
        super(LinearModel, self).__init__()
        self.encoder = encoder
        if hasattr(self.encoder, "inplanes"):
            self.features_dim = self.encoder.inplanes
        else:
            self.features_dim = self.encoder.num_features
        self.classifier = nn.Linear(self.features_dim, num_classes)  # type: ignore

        # training related
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        assert scheduler_interval in ["step", "epoch"]
        self.scheduler_interval = scheduler_interval
        self.lr_decay_steps = lr_decay_steps
        self.no_channel_last = no_channel_last

        self._num_training_steps = None
        self.tasks = tasks
        self.num_tasks = num_tasks
        # all the other parameters
        self.extra_args = kwargs

        for param in self.backbone.parameters():
            param.requires_grad = False

        if scheduler_interval == "step":
            warnings.warn(
                f"Using scheduler_interval={scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

        # can provide up to ~20% speed up
        if not no_channel_last:
            self = self.to(memory_format=torch.channels_last)

    @property
    def num_training_steps(self) -> int:
        """Compute the number of training steps for each epoch."""

        if self._num_training_steps is None:
            try:
                dataset = self.extra_args.get("dataset", None)
                if dataset not in ["cifar10", "cifar100", "stl10"]:
                    data_path = self.extra_args.get("train_data_path", "./train")
                else:
                    data_path = None

                no_labels = self.extra_args.get("no_labels", False)
                data_fraction = self.extra_args.get("data_fraction", -1.0)
                data_format = self.extra_args.get("data_format", "image_folder")
                dataset_size = compute_dataset_size(
                    dataset=dataset,
                    data_path=data_path,
                    data_format=data_format,
                    train=True,
                    no_labels=no_labels,
                    data_fraction=data_fraction,
                )
            except:
                raise RuntimeError(
                    "Please pass 'dataset' or 'train_data_path' as parameters to the model."
                )

            dataset_size = self.trainer.limit_train_batches * dataset_size

            num_devices = self.trainer.num_devices
            num_nodes = self.trainer.num_nodes
            effective_batch_size = (
                self.batch_size * self.trainer.accumulate_grad_batches * num_devices * num_nodes
            )
            self._num_training_steps = dataset_size // effective_batch_size

        return self._num_training_steps

    def forward(self, x):
        if not self.no_channel_last:
            x = x.to(memory_format=torch.channels_last)
        with torch.no_grad():
            feats = self.encoder(x)
        logits = self.fc(feats)
        # return {"logits": logits, "feats": feats}
        return logits

    def eval_acc_step(self, logits, targets):
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))
        return {
            "acc1": acc1,
            "acc5": acc5,
        }

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        x, targets = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, targets)

        acc_metrics = self.eval_acc_step(logits, targets)
        metrics = {
            "train_loss": loss,
            "train_acc1": acc_metrics["acc1"],
            "train_acc5": acc_metrics["acc5"],
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return {'loss': loss}

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, targets = batch
        batch_size = targets.size(0)
        logits = self.forward(x)
        loss = F.cross_entropy(logits, targets)
        acc_metrics = self.eval_acc_step(logits, targets)
        metrics = {
            "batch_size": batch_size,
            "targets": targets,
            "logits": logits,
            "val_loss": loss,
            "val_acc1": acc_metrics["acc1"],
            "val_acc5": acc_metrics["acc5"],
        }

        if self.split_strategy == "domain" and len(batch) == 3:
            metrics["domains"] = batch[0]

        return {**metrics}

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}

        if not self.trainer.sanity_checking:
            preds = torch.cat([o["logits"].max(-1)[1] for o in outs]).cpu().numpy()
            targets = torch.cat([o["targets"] for o in outs]).cpu().numpy()
            mask_correct = preds == targets
            assert self.tasks is not None
            for task_idx, task in enumerate(self.tasks):
                mask_task = np.isin(targets, np.array(task))
                correct_task = np.logical_and(mask_task, mask_correct).sum()
                log[f"val_acc1_task{task_idx}"] = correct_task / mask_task.sum()

        self.log_dict(log, sync_dist=True)

    @property
    def current_task_idx(self) -> int:
        return getattr(self, "_current_task_idx", None)

    @current_task_idx.setter
    def current_task_idx(self, new_task):
        if hasattr(self, "_current_task_idx"):
            assert new_task >= self._current_task_idx
        self._current_task_idx = new_task

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {
                "name": "encoder",
                "params": self.encoder.parameters(),
                "lr": 0,
                "weight_decay": 0,
            },
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.lr,
                "weight_decay": 0,
            },
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")

        # create optimizer
        optimizer = optimizer(
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )
        # optionally wrap with lars
        if self.lars:
            optimizer = LARSWrapper(
                optimizer,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            )

        if self.scheduler == "none":
            return optimizer
        else:
            if self.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs,
                    max_epochs=self.max_epochs,
                    warmup_start_lr=self.warmup_start_lr,
                    eta_min=self.min_lr,
                )
            elif self.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, self.max_epochs, eta_min=self.min_lr)
            elif self.scheduler == "step":
                scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
            else:
                raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

            if idxs_no_scheduler:
                partial_fn = partial(
                    static_lr,
                    get_lr=scheduler.get_lr,
                    param_group_indexes=idxs_no_scheduler,
                    lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
                )
                scheduler.get_lr = partial_fn

            return [optimizer], [scheduler]

