import torch
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from utils.dataset_utils import get_dataset_joint, get_pretrained_dataset, split_dataset
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.encoder_utils_lwf import get_pretrained_encoder
from utils.args_utils import parse_args_cpn
from models.linear_CJT import MLP


def main():
    seed_everything(5)
    args = parse_args_cpn()
    num_gpus = [0]

    classes_order = torch.tensor(
        [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28,
         53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97,
         2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7,
         63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33])
    tasks_initial = classes_order[:int(args.num_classes / 2)].chunk(1)
    tasks_incremental = classes_order[int(args.num_classes / 2):args.num_classes].chunk(args.num_tasks)
    tasks = tasks_initial + tasks_incremental
    train_dataset, test_dataset = get_dataset_joint(dataset=args.dataset, data_path=args.data_path)

    for task_idx in range(0, args.num_tasks + 1):
        filename = f"{args.pretrained_model}/task_{task_idx}_pretrain_0samples_cifar100_0_{int(100/args.num_tasks)}_1993_resnet50.pt"
        if "cifar" in args.dataset:
            encoder = get_pretrained_encoder(filename, cifar=True)
            encoder.fc = torch.nn.Identity()
        else:
            encoder = get_pretrained_encoder(filename, cifar=False)
            encoder.fc = torch.nn.Identity()

        model = MLP(**args.__dict__)
        model.encoder = encoder

        wandb_logger = WandbLogger(
            name=f"LWF-CJT-{args.dataset}-{args.pretrained_method}-{args.num_tasks}tasks-steps{task_idx}",
            project=args.project,
            entity=args.entity,
            offline=False,
        )
        if args == 0:
            wandb_logger.log_hyperparams(args)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        train_dataset_task = split_dataset(
            train_dataset,
            tasks=tasks,
            task_idx=list(range(task_idx + 1)),
        )
        test_dataset_task = split_dataset(
            test_dataset,
            tasks=tasks,
            task_idx=list(range(task_idx + 1)),
        )
        train_loader = DataLoader(train_dataset_task, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset_task, batch_size=args.batch_size,
                                 shuffle=True, num_workers=8, pin_memory=True)
        trainer = pl.Trainer(
            gpus=num_gpus,
            max_epochs=args.epochs,
            accumulate_grad_batches=1,
            sync_batchnorm=True,
            # accelerator='gpu',
            logger=wandb_logger,
            enable_checkpointing=False,
            precision=16,
            callbacks=[lr_monitor]

        )
        trainer.fit(model, train_loader, test_loader)
        trainer.test(model, test_loader)
        wandb.finish()


if __name__ == '__main__':
    main()
