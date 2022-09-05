import torch
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from utils.dataset_utils import get_dataset, get_pretrained_dataset, split_dataset
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.encoder_utils import get_pretrained_encoder
from utils.args_utils import parse_args_cpn
from models.linear import MLP


def main():
    seed_everything(5)
    args = parse_args_cpn()
    num_gpus = [0]
    encoder = get_pretrained_encoder(args.pretrained_model, cifar=False)
    model = MLP(**args.__dict__)
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    from torchvision import datasets, transforms
    cifar_transforms = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_dataset = datasets.CIFAR100(root=args.data_path, train=True,
                                      transform=cifar_transforms,
                                      download=True)
    test_dataset = datasets.CIFAR100(root=args.data_path, train=False,
                                     transform=cifar_transforms,
                                     download=True)
    wandb_logger = WandbLogger(
        name=f"{args.perfix}{args.dataset}-{args.pretrained_method}-linear-eval",
        project=args.project,
        entity=args.entity,
        offline=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    train_dataset_pretrained, test_dataset_pretrained, cpn_means = get_pretrained_dataset(
        encoder=encoder,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tau=args.tau,
        return_means=True)

    train_loader = DataLoader(train_dataset_pretrained, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset_pretrained, batch_size=64, shuffle=True)
    trainer = pl.Trainer(
        gpus=num_gpus,
        max_epochs=args.epochs,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        accelerator='ddp',
        logger=wandb_logger,
        checkpoint_callback=False,
        precision=16,
        callbacks=[lr_monitor]

    )
    batch = next(iter(train_loader))
    x, targets = batch
    print(x.shape, targets.shape)
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
