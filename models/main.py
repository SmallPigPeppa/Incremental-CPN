import torch
import torch.nn as nn
import ds_imagenet100_exclude as ds
import pytorch_lightning as pl
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
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
        # scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=EPOCHS)
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

    x_train = 10. * np.load('../simclr32_3/data_pretrained/imagenet100_exclude/x_train.npy')
    x_test = 10. * np.load('../simclr32_3/data_pretrained/imagenet100_exclude/x_test.npy')
    y_train = np.load('../simclr32_3/data_pretrained/imagenet100_exclude/y_train.npy').astype(int)
    y_test = np.load('../simclr32_3/data_pretrained/imagenet100_exclude/y_test.npy').astype(int)
    print(y_train)
    print(y_test)

    EPOCHS = 100
    LAMBDA1= 0.
    # LAMBDA2 = 0.15*50
    LAMBDA2 = 0.
    LR = 6.0
    BATCH_SIZE = 1024
    NUM_GPUS = [5,6]
    NUM_WORKERS = 1
    INCREMENTAL_N=5
    mds = ds.IncrementalDatasetCifar100()
    train_ds = mds.get_train_set(previous_classes=range(50), incremental_classes=range(50), discout_ratio=0.)
    test_ds = mds.get_test_set(previous_classes=range(50), incremental_classes=range(50))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    mmodel = IncrementalPT(cn=50, dim_feature=2048, means=mds.get_means(classes=range(50)))
    trainer = pl.Trainer(
        gpus=NUM_GPUS,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        accelerator='ddp',
        logger = TensorBoardLogger(f"./logs/", name=f"imgnet_{int(50/INCREMENTAL_N)}steps_{0}"),
             checkpoint_callback = False
    )
    trainer.fit(mmodel, train_loader, test_loader)
    # # trainer.test(mmodel, test_loader)
    for i in range(int(50 / INCREMENTAL_N)):
        # EPOCHS = 160
        LAMBDA1 = 0.
        LAMBDA2 = 0.
        means = mds.get_means(classes=range(50 + i * INCREMENTAL_N, 50 +(i+1)*INCREMENTAL_N ))
        mmodel.upadate_w(incremental_cn=INCREMENTAL_N, means=means)
        train_ds = mds.get_train_set(previous_classes=range(0, 50 + i * INCREMENTAL_N),
                                     incremental_classes=range(50 + i * INCREMENTAL_N, 50 +(i+1)*INCREMENTAL_N ), discout_ratio=0.)
        test_ds = mds.get_test_set(previous_classes=range(0, 50 + i * INCREMENTAL_N),
                                   incremental_classes=range(50 + i * INCREMENTAL_N, 50 +(i+1)*INCREMENTAL_N ))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
        trainer = pl.Trainer(
            gpus=NUM_GPUS,
            max_epochs=EPOCHS,
            accumulate_grad_batches=1,
            sync_batchnorm=True,
            accelerator='ddp',
            logger=TensorBoardLogger(f"./logs/", name=f"imgnet_{int(50/INCREMENTAL_N)}steps_{i+1}"),
            checkpoint_callback=False
        )
        trainer.fit(mmodel, train_loader, test_loader)
