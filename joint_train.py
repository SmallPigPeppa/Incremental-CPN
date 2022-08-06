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
    assert args.num_classes % args.num_tasks == 0
    tasks = torch.randperm(args.num_classes).chunk(args.num_tasks)

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
        model = LinearModel(encoder, tasks=tasks, **args.__dict__)
    elif args.method == 'cpn':
        model = CPNModule(encoder, tasks=tasks, pl_lamda=args.pl_lambda, **args.__dict__)

    make_contiguous(model)

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
    ckpt_path, wandb_run_id = None, None
    if args.auto_resume and args.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(args.checkpoint_dir, "linear"),
            max_hours=args.auto_resumer_max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(args)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    callbacks = []

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, "linear"),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

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
        trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
