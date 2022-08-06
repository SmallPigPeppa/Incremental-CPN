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
    def __init__(self,features_dim,num_classes):
        super(PrototypeClassifier,self).__init__()
        self.features_dim = features_dim
        self.num_calsses=num_classes
        # self.w = nn.Parameter(torch.tensor(means))
        self.prototype = nn.Parameter(torch.random(features_dim,num_classes))

    def forward(self, x):
        x = x.reshape(-1, 1, self.dim_feature)
        d = torch.pow(x - self.prototype, 2)
        d = torch.sum(d, dim=2)
        return d

    def current_task_initial(self,means,current_tasks=list(range(10))):
        if means is not None:
            nn.init.constant_(self.prototype.weight.data[current_tasks,:], means)
        disable_backward=[i for i in range(self.num_calsses) if i not in current_tasks]
        self.prototype.weight[disable_backward,:].requires_grad = False

class CPNModule(LinearModel):
    def __init__(self,current_tasks,**kwargs):
        super(LinearModel, self).__init__(**kwargs)
        self.current_tasks=current_tasks
        self.classifier=PrototypeClassifier(self.features_dim,self.num_classes)
