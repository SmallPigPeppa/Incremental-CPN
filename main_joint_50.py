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
    # cifar_transforms = transforms.Compose(
    #     [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
    # train_dataset = datasets.CIFAR100(root=args.data_path, train=True,
    #                                   transform=cifar_transforms,
    #                                   download=True)
    # test_dataset = datasets.CIFAR100(root=args.data_path, train=False,
    #                                  transform=cifar_transforms,
    #                                  download=True)
    import os
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_path = os.path.join(args.data_path, "imagenet100")
    imagenet_tansforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"),
                                         transform=imagenet_tansforms)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"),
                                        transform=imagenet_tansforms)
    classes_order = torch.tensor(list(range(args.num_classes)))
    classes_order = torch.tensor(
        [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28,
         53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97,
         2,
         64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63,
         75,
         5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33])
    tasks_initial = classes_order[:int(args.num_classes / 2)].chunk(1)
    tasks_incremental = classes_order[int(args.num_classes / 2):args.num_classes].chunk(args.num_tasks)
    tasks = tasks_initial + tasks_incremental
    train_dataset = split_dataset(
        train_dataset,
        tasks=tasks,
        task_idx=[0],
    )
    test_dataset = split_dataset(
        test_dataset,
        tasks=tasks,
        task_idx=list(range(0 + 1)),
    )
    wandb_logger = WandbLogger(
        name=f"{args.perfix}{args.dataset}-{args.pretrained_method}-linear-eval-step0",
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
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
