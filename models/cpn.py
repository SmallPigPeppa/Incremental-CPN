import torch
import torch.nn as nn

import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
from utils.metrics import accuracy_at_k, weighted_mean
import warnings
import numpy as np
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from utils.lars import LARS
from utils.misc import compute_dataset_size
from argparse import ArgumentParser
from models.linear import LinearModel


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


class CPNModule(LinearModel):
    def __init__(self, current_tasks, pl_lambda, **kwargs):
        super().__init__(**kwargs)
        self.current_tasks = current_tasks
        self.classifier = PrototypeClassifier(self.features_dim, self.num_classes)
        self.pl_lambda = pl_lambda

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR and supervised SimCLR reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """
        out = super().training_step(batch, batch_idx)
        squared_Euclidean_distance=-1.*out["logits"]
        prototype_loss = torch.index_select(squared_Euclidean_distance, dim=1, index=out["targets"])
        prototype_loss = torch.diagonal(prototype_loss)
        prototype_loss = torch.mean(prototype_loss)

        metrics = {
            "prototype_loss": prototype_loss,
            "pl_lambda":self.pl_lambda
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        out.update({"loss": out["loss"] + self.pl_lambda * prototype_loss})
        return out

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        super().on_train_start()
        self.classifier.incremental_initial(current_tasks=self.current_tasks)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic momentum arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parent_parser = super(CPNModule, CPNModule).add_model_specific_args(
            parent_parser
        )
        parser = parent_parser.add_argument_group("cpn")
        # momentum settings
        parser.add_argument("--pl_lambda", default=0.01, type=float)

        return parent_parser


if __name__ == '__main__':
    a = PrototypeClassifier(features_dim=10, num_classes=3)
    # print(a)
    # print(a.prototypes[0])
    a.incremental_initial(current_tasks=[1], means=torch.ones([10]))
    x = torch.rand([5, 10])
    x = a(x)
    # print(x)
    loss = sum(sum(x))
    loss.backward()
    # x.backward()
    for x in a.prototypes:
        print(x)
        print(x.grad)
