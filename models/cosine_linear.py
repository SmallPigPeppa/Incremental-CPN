import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR



class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, input):
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.bias is not None:
            cosine = cosine + self.bias
        return cosine


class MLP(pl.LightningModule):
    def __init__(self, dim_feature, num_classes, pl_lambda, lr, epochs, warmup_epochs, **kwargs):
        super(MLP, self).__init__()
        self.dim_feature = dim_feature
        self.num_calsses = num_classes
        self.pl_lambda = pl_lambda
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.extra_args = kwargs
        self.model = CosineLinear(dim_feature, num_classes)

    def task_initial(self, current_tasks, means=None):
        # if means is not None:
        #     for i in current_tasks:
        #         self.prototypes[i].data = torch.nn.Parameter((means[str(i)]).reshape(1, -1))1
        # no_grad_idx = [i for i in range(self.num_calsses) if i not in current_tasks]
        # for i in no_grad_idx:
        #     self.prototypes[i].requires_grad = False
        # for i in current_tasks:
        #     self.prototypes[i].requires_grad = True
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=self.warmup_epochs,
                                                  max_epochs=self.epochs)
        return [optimizer], [scheduler]

    def forward(self, x):
        out = self.model(x)
        return out

    def share_step(self, batch, batch_idx):
        x, targets = batch
        logits = self.forward(x)
        # ce loss
        ce_loss = F.cross_entropy(logits, targets)
        # acc
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == targets) / targets.shape[0]
        return {"acc": acc, "loss": ce_loss}

    def training_step(self, batch, batch_idx):
        out = self.share_step(batch, batch_idx)
        log_dict = {"train_" + k: v for k, v in out.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return out

    def validation_step(self, batch, batch_idx):
        out = self.share_step(batch, batch_idx)
        log_dict = {"val_" + k: v for k, v in out.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return out

    def test_step(self, batch, batch_idx):
        out = self.share_step(batch, batch_idx)
        log_dict = {"test_" + k: v for k, v in out.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return out
