import os
import types
import warnings
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchvision.models import resnet18, resnet50
from pytorch_lightning.strategies.ddp import DDPStrategy
from args.setup import parse_args_linear
from utils.auto_resumer import AutoResumer
import wandb

try:
    from cassle.methods.dali import ClassificationABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True
from models.linear import LinearModel
from models.cpn import CPNModule
from utils.misc import make_contiguous
from utils.classification_dataloader import prepare_data
from utils.checkpointer import Checkpointer

try:
    from utils.dali_dataloader import ClassificationDALIDataModule
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True
from args.setup import parse_args_linear


def main():
    seed_everything(5)

    args = parse_args_linear()

    # split classes into tasks
    tasks_initial = torch.tensor(list(range(int(args.num_classes / 2)))).chunk(1)
    tasks_incremental = torch.tensor(list(range(int(args.num_classes / 2), args.num_classes))).chunk(
        args.num_tasks - 1)
    tasks = tasks_initial + tasks_incremental

    if args.encoder == "resnet18":
        encoder = resnet18()
    elif args.encoder == "resnet50":
        encoder = resnet50()
    else:
        raise ValueError("Only [resnet18, resnet50] are currently supported.")

    cifar = True if 'cifar' in args.dataset else False
    if cifar:
        encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        encoder.maxpool = nn.Identity()

    encoder.fc = nn.Identity()
    assert (
            args.pretrained_feature_extractor.endswith(".ckpt")
            or args.pretrained_feature_extractor.endswith(".pth")
            or args.pretrained_feature_extractor.endswith(".pt")
    )
    ckpt_path = args.pretrained_feature_extractor

    state = torch.load(ckpt_path)["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
            warnings.warn(
                "You are using an older checkpoint. Use a new one as some issues might arrise."
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    encoder.load_state_dict(state, strict=False)

    print(f"Loaded {ckpt_path}")

    del args.encoder

    if args.method == 'linear':
        model = LinearModel(encoder=encoder, tasks=tasks, **args.__dict__)
    elif args.method == 'cpn':
        current_tasks = list(range(100))
        # current_tasks = tasks[task_idx]
        model = CPNModule(encoder=encoder, current_tasks=current_tasks, pl_lamda=args.pl_lambda, tasks=tasks,
                          **args.__dict__)
        from utils.get_cpn_means import get_means
        train_loader, val_loader = prepare_data(
            args.dataset,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            data_format=args.data_format,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tasks=None
        )
        cpn_means = get_means(encoder=encoder, train_loader=train_loader, classes=list(range(100)))

    make_contiguous(model)

    for task_idx in range(args.num_tasks):
        print(f"################## start task {task_idx} ##################")

        if args.method == 'cpn':
            # change current_tasks
            current_tasks = tasks[task_idx]
            model.classifier.incremental_initial(means=cpn_means[current_tasks], current_tasks=current_tasks)

        if args.data_format == "dali":
            val_data_format = "image_folder"
        else:
            val_data_format = args.data_format
        train_loader, val_loader = prepare_data(
            args.dataset,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            data_format=val_data_format,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            task_idx=task_idx,
            tasks=tasks
        )

        if args.data_format == "dali":
            assert (
                _dali_avaliable
            ), "Dali is not currently avaiable, please install it first with [dali]."

            dali_datamodule = ClassificationDALIDataModule(
                dataset=args.dataset,
                train_data_path=args.train_data_path,
                val_data_path=args.val_data_path,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                data_fraction=args.data_fraction,
                dali_device=args.dali_device,
            )

            # use normal torchvision dataloader for validation to save memory
            dali_datamodule.val_dataloader = lambda: val_loader

        # 1.7 will deprecate resume_from_checkpoint, but for the moment
        # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
        callbacks = []
        # wandb logging
        if args.wandb:
            wandb_logger = WandbLogger(
                name=f"{args.name}-task{task_idx}",
                project=args.project,
                entity=args.entity,
                offline=args.offline,
            )
            # wandb_logger.log_hyperparams(args)

            # lr logging
            lr_monitor = LearningRateMonitor(logging_interval="step")
            callbacks.append(lr_monitor)

        trainer = Trainer.from_argparse_args(
            args,
            logger=wandb_logger if args.wandb else None,
            callbacks=callbacks,
            enable_checkpointing=False,
            strategy=DDPStrategy(find_unused_parameters=False)
            if args.strategy == "ddp"
            else args.strategy,
        )

        # fix for incompatibility with nvidia-dali and pytorch lightning
        # with dali 1.15 (this will be fixed on 1.16)
        # https://github.com/Lightning-AI/lightning/issues/12956
        try:
            from pytorch_lightning.loops import FitLoop

            class WorkaroundFitLoop(FitLoop):
                @property
                def prefetch_batches(self) -> int:
                    return 1

            trainer.fit_loop = WorkaroundFitLoop(
                trainer.fit_loop.min_epochs, trainer.fit_loop.max_epochs
            )
        except:
            pass

        if args.data_format == "dali":
            trainer.fit(model, ckpt_path=None, datamodule=dali_datamodule)
        else:
            trainer.fit(model, train_loader, val_loader, ckpt_path=None)

        wandb.finish()

if __name__ == "__main__":
    main()
