import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from utils.dataset_utils import split_dataset
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.args_utils import parse_args_cpn
from models.icpn import IncrementalCPN
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os


def main():
    seed_everything(5)
    args = parse_args_cpn()
    num_gpus = [0]
    model = IncrementalCPN(**args.__dict__)

    classes_order = torch.tensor(
        [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28,
         53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97,
         2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7,
         63,
         75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33])
    # classes_order = torch.randperm(num_classes)
    # classes_order = torch.tensor(list(range(args.num_classes)))
    tasks_initial = classes_order[:int(args.num_classes / 2)].chunk(1)
    tasks_incremental = classes_order[int(args.num_classes / 2):args.num_classes].chunk(args.num_tasks)
    tasks = tasks_initial + tasks_incremental
    x_train = np.load(os.path.join(args.data_path, args.dataset, "x_train.npy"))
    x_test = np.load(os.path.join(args.data_path, args.dataset, "x_test.npy"))
    y_train = np.load(os.path.join(args.data_path, args.dataset, "y_train.npy"))
    y_test = np.load(os.path.join(args.data_path, args.dataset, "y_test.npy"))
    train_dataset_pretrained = TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset_pretrained = TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long))
    for task_idx in range(0, args.num_tasks + 1):
        train_dataset_task = split_dataset(
            train_dataset_pretrained,
            tasks=tasks,
            task_idx=[task_idx],
        )
        test_dataset_task = split_dataset(
            test_dataset_pretrained,
            tasks=tasks,
            task_idx=list(range(task_idx + 1)),
        )
        train_loader = DataLoader(train_dataset_task, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset_task, batch_size=64, shuffle=True)
        wandb_logger = WandbLogger(
            name=f"{args.perfix}{args.dataset}-{args.pretrained_method}-lambda{args.pl_lambda}-{args.num_tasks}tasks-steps{task_idx}",
            project=args.project,
            entity=args.entity,
            offline=False,
        )
        if args == 0:
            wandb_logger.log_hyperparams(args)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        if args.cpn_initial == "means":
            # model.task_initial(current_tasks=tasks[task_idx], means=cpn_means)
            pass
        else:
            model.task_initial(current_tasks=tasks[task_idx])
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
        wandb.finish()

if __name__ == '__main__':
    main()
