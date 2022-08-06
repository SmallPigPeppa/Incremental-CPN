import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import multiprocessing
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
import torchvision

class TopModel(pl.LightningModule):
    def __init__(self, cn):
        super(TopModel, self).__init__()
        self.cn=cn
        self.encoder = getattr(torchvision.models, 'resnet50')(pretrained=False)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.feature_size = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.fc=nn.Linear(self.feature_size, cn)
    def forward(self, x):
        x=self.encoder(x)
        logits=self.fc(x)
        return logits
    def training_step(self, batch, batch_idx):
        x, targets = batch
        logits = self.forward(x)
        ce_loss = F.cross_entropy(logits, targets)
        self.log("c", ce_loss, prog_bar=True)
        # acc
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == targets) / targets.shape[0]
        self.log('acc', acc, prog_bar=True)
        return ce_loss

    def configure_optimizers(self):
        # opt=torch.optim.AdamW(self.parameters(), lr=LR,weight_decay=0.1)
        # opt=torch.optim.Adam(self.parameters(), lr=LR)
        opt = torch.optim.SGD(self.parameters(), lr=LR, momentum=0.9,weight_decay = 5e-4)
        scheduler=LinearWarmupCosineAnnealingLR(optimizer=opt, warmup_epochs=10, max_epochs=EPOCHS)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100,200,300], gamma=0.1)
        return {'optimizer':opt,"lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "metric_to_track",
            }}


    def evaluate(self, batch, stage=None):
        x, targets = batch
        logits = self(x)
        ce_loss = F.cross_entropy(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == targets) / targets.shape[0]
        if stage:
            self.log('cv', ce_loss, prog_bar=True)
            self.log('accv', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')
    def update_fc(self,cn):
        self.cn=cn
        self.fc=nn.Linear(self.feature_size, cn)


train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ]
)
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ]
)


if __name__ == '__main__':
    EPOCHS =400
    BATCH_SIZE = 512
    LR = 0.1
    NUM_GPUS = [7]
    NUM_CLASSES = 10
    # NUM_WORKERS = multiprocessing.cpu_count()
    NUM_WORKERS = 1
    train_ds_all = torchvision.datasets.CIFAR100(root='/lustre/home/wzliu/datasets', train=True,download=True, transform= train_transforms)
    idx_train=[]
    for i in range(len(train_ds_all)):
        if train_ds_all[i][1] in list(range(50)):
            idx_train.append(i)
    train_ds = torch.utils.data.Subset(train_ds_all, idx_train)
    test_ds_all = torchvision.datasets.CIFAR100(root='/lustre/home/wzliu/datasets', train=False,download=True, transform= test_transforms)
    idx_test=[]
    for i in range(len(test_ds_all)):
        if test_ds_all[i][1] in list(range(50)):
            idx_test.append(i)
    test_ds = torch.utils.data.Subset(test_ds_all, idx_test)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    mmodel = TopModel(cn=50)
    trainer = pl.Trainer(
        gpus=NUM_GPUS,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        accelerator='ddp',
        logger=TensorBoardLogger("./logs/", name="cifar_semitop_0"),
        checkpoint_callback=False,
        precision=16
    )
    #        precision = 16,
    trainer.fit(mmodel, train_loader, test_loader)
    for task in range(10):
        mmodel.update_fc(cn=55+task*5)
        # EPOCHS = 100
        idx_train = []
        print("prepare datasets ...")
        for i in range(len(train_ds_all)):
            if train_ds_all[i][1] in list(range(55+task*5)):
                idx_train.append(i)
        train_ds = torch.utils.data.Subset(train_ds_all, idx_train)
        idx_test = []
        for i in range(len(test_ds_all)):
            if test_ds_all[i][1] in list(range(55+task*5)):
                idx_test.append(i)
        print("finish prepare datasets ...")
        test_ds = torch.utils.data.Subset(test_ds_all, idx_test)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
        trainer = pl.Trainer(
            gpus=NUM_GPUS,
            max_epochs=EPOCHS,
            accumulate_grad_batches=1,
            sync_batchnorm=True,
            accelerator='ddp',
            logger=TensorBoardLogger(f"./logs/", name=f"cifar_semitop_{task+1}"),
            checkpoint_callback=False,
            precision=16
        )
        trainer.fit(mmodel, train_loader, test_loader)
