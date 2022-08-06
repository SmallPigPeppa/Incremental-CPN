import torch
import torch.nn as nn
# import pytorch_lightning as pl
# import torch.nn.functional as F
# from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
# from utils.metrics import accuracy_at_k, weighted_mean
# import warnings
# import numpy as np
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
# from utils.lars import LARS
# from utils.misc import compute_dataset_size
# from argparse import ArgumentParser
# from models.linear import LinearModel
class PrototypeClassifier(nn.Module):
    def __init__(self,features_dim,num_classes):
        super(PrototypeClassifier,self).__init__()
        self.features_dim = features_dim
        self.num_calsses=num_classes
        # self.w = nn.Parameter(torch.tensor(means))
        self.prototype = nn.Parameter(torch.rand([num_classes,features_dim]))
        self.no_grad_idx=None

    def forward(self, x):
        if self.no_grad_idx :
            with torch.no_grad():
                no_grad_prototype = self.prototype[self.no_grad_idx, :]
            tmp_prototype=self.prototype
            # tmp_prototype[self.no_grad_idx]=no_grad_prototype
            tmp_prototype.index_fill_(1, self.no_grad_idx,no_grad_prototype)

        x = x.reshape(-1, 1, self.features_dim)
        d = torch.pow(x - self.prototype, 2)
        d = torch.sum(d, dim=2)
        return d

    def incremental_initial(self,means=None,current_tasks=list(range(10))):
        if means is not None:
            nn.init.constant_(self.prototype.weight.data[current_tasks,:], means)
        self.no_grad_idx=[i for i in range(self.num_calsses) if i not in current_tasks]

# class CPNModule(LinearModel):
#     def __init__(self,current_tasks,**kwargs):
#         super(LinearModel, self).__init__(**kwargs)
#         self.current_tasks=current_tasks
#         self.classifier=PrototypeClassifier(self.features_dim,self.num_classes)


if __name__=='__main__':
    a=PrototypeClassifier(features_dim=10,num_classes=3)
    a.incremental_initial(current_tasks=[1])
    x=torch.rand([5,10])
    x=a(x)
    print(x)
    loss=sum(sum(x))
    loss.backward()
    # x.backward()
    print(a.prototype.grad)