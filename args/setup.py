# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse

import pytorch_lightning as pl
from args.dataset import (
    augmentations_args,
    custom_dataset_args,
    dataset_args,
    linear_augmentations_args,
)
from args.utils import additional_setup_linear
from utils.auto_resumer import AutoResumer
from utils.checkpointer import Checkpointer


try:
    from utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

try:
    from utils.dali_dataloader import ClassificationDALIDataModule, PretrainDALIDataModule
except ImportError:
    _dali_available = False
else:
    _dali_available = True




def parse_args_linear() -> argparse.Namespace:
    """Parses feature extractor, dataset, pytorch lightning, linear eval specific and additional args.

    First adds an arg for the pretrained feature extractor, then adds dataset, pytorch lightning
    and linear eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_feature_extractor", type=str, required=True)
    parser.add_argument("--pretrain_method", type=str, default=None)

    # add shared arguments
    dataset_args(parser)
    linear_augmentations_args(parser)
    custom_dataset_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # linear model
    from models.linear import LinearModel
    parser =LinearModel.add_model_specific_args(parser)

    # THIS LINE IS KEY TO PULL WANDB AND SAVE_CHECKPOINT
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--auto_resume", action="store_true")
    temp_args, _ = parser.parse_known_args()

    # optionally add checkpointer
    if temp_args.save_checkpoint:
        parser = Checkpointer.add_checkpointer_args(parser)

    if temp_args.auto_resume:
        parser = AutoResumer.add_autoresumer_args(parser)

    if _dali_available and temp_args.data_format == "dali":
        parser = ClassificationDALIDataModule.add_dali_args(parser)


    # incremental
    parser.add_argument("--num_tasks", type=int,default=5)
    # parse args
    args = parser.parse_args()
    additional_setup_linear(args)

    return args

