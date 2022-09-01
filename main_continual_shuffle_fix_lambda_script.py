import torch
import pytorch_lightning as pl
import wandb
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from utils.dataset_utils import get_dataset
from utils import get_pretrained_dataset, get_pretrained_encoder, split_dataset
from args import parse_args_cpn
from models.icpn import IncrementalCPN


def main():
    args = parse_args_cpn()
    num_classes = 100
    num_gpus = [0, 1]

    encoder = get_pretrained_encoder(args.pretrained_model)
    model = IncrementalCPN(**args.__dict__)

    classes_order = torch.tensor(list(range(num_classes)))
    # classes_order = torch.randperm(num_classes)
    tasks_initial = classes_order[:int(num_classes / 2)].chunk(1)
    tasks_incremental = classes_order[int(num_classes / 2):num_classes].chunk(args.num_tasks)
    tasks = tasks_initial + tasks_incremental
    train_dataset, test_dataset = get_dataset(dataset=args.dataset, data_path=args.data_path)

    for task_idx in range(0, args.num_tasks + 1):
        model.task_initial(current_tasks=tasks[task_idx])
        wandb_logger = WandbLogger(
            name=f"{args.dataset}-{args.pretrained_method}-{args.num_tasks}tasks-steps{task_idx}-lambda{args.pl_lambda}",
            project="Incremental-CPN-v6",
            entity="pigpeppa",
            offline=False,
        )
        train_dataset_task = split_dataset(
            train_dataset,
            tasks=tasks,
            task_idx=[task_idx],
        )
        test_dataset_task = split_dataset(
            test_dataset,
            tasks=tasks,
            task_idx=list(range(task_idx + 1)),
        )
        train_dataset_task, test_dataset_task = get_pretrained_dataset(encoder=encoder,
                                                                       train_dataset=train_dataset_task,
                                                                       test_dataset=test_dataset_task)
        train_loader = DataLoader(train_dataset_task, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset_task, batch_size=64, shuffle=True)

        trainer = pl.Trainer(
            gpus=num_gpus,
            max_epochs=args.epochs,
            accumulate_grad_batches=1,
            sync_batchnorm=True,
            accelerator='ddp',
            logger=wandb_logger,
            checkpoint_callback=False,
            precision=16,

        )
        trainer.fit(model, train_loader, test_loader)
        wandb.finish()


if __name__ == '__main__':
    main()
