import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class MLP(pl.LightningModule):
    def __init__(self, dim_feature, num_classes, lr, epochs, warmup_epochs, lambda1, lambda2=0., **kwargs):
        super(MLP, self).__init__()
        self.dim_feature = dim_feature
        self.num_calsses = num_classes
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.extra_args = kwargs
        self.prototypes = nn.ParameterList(
            [nn.Parameter(torch.randn(1, self.dim_feature)) for i in range(num_classes)])

    def task_initial(self, current_tasks, means=None):
        if means is not None:
            for i in current_tasks:
                self.prototypes[i].data = torch.nn.Parameter((means[str(i)]).reshape(1, -1))
        no_grad_idx = [i for i in range(self.num_calsses) if i not in current_tasks]
        for i in no_grad_idx:
            self.prototypes[i].requires_grad = False
        for i in current_tasks:
            self.prototypes[i].requires_grad = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=self.warmup_epochs,
                                                  max_epochs=self.epochs)
        return [optimizer], [scheduler]

    # def forward(self, x):
    #     x = x.reshape(-1, 1, self.dim_feature)
    #     prototypes_list = [i for i in self.prototypes]
    #     d = torch.pow(x - torch.cat(prototypes_list), 2)
    #     d = torch.sum(d, dim=2)
    #     return d

    def forward(self, x):
        x = x.reshape(-1, 1, self.dim_feature)
        prototypes_list = [i for i in self.prototypes]
        prototypes = torch.cat(prototypes_list)
        import pdb;pdb.set_trace()
        cosine = F.linear(F.normalize(x, p=2, dim=1), F.normalize(prototypes, p=2, dim=1))
        return cosine

    def share_step(self, batch, batch_idx):
        x, targets = batch
        d = self.forward(x)
        logits = -1. * d
        # ce loss
        ce_loss = F.cross_entropy(logits, targets)
        # pl loss
        pl_loss = torch.index_select(d, dim=1, index=targets)
        pl_loss = torch.diagonal(pl_loss)
        pl_loss = torch.mean(pl_loss)
        # pl cosine loss
        prototypes_list = [i for i in self.prototypes]
        w = torch.cat(prototypes_list)
        c = torch.index_select(w, dim=0, index=targets)
        cos_matrix = F.normalize(x, p=2, dim=1) * F.normalize(c, p=2, dim=1)
        pl_cos_loss = -1. * torch.sum(cos_matrix)
        # all loss
        loss = ce_loss + pl_loss * self.lambda1 + pl_cos_loss * self.lambda2
        # acc
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == targets) / targets.shape[0]
        return {"ce_loss": ce_loss, "pl_loss": pl_loss, "pl_cosine_loss": pl_cos_loss, "acc": acc, "loss": loss}

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
