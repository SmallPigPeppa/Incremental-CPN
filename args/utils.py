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

import os
import warnings
from argparse import Namespace
from contextlib import suppress

N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "imagenet": 1000,
    "imagenet100": 100,
}


def additional_setup_linear(args: Namespace):
    """Provides final setup for linear evaluation to non-user given parameters by changing args.

    Parsers arguments to extract the number of classes of a dataset, correctly parse gpus, identify
    if a cifar dataset is being used and adjust the lr.

    Args:
        args: Namespace object that needs to contain, at least:
        - dataset: dataset name.
        - optimizer: optimizer name being used.
        - gpus: list of gpus to use.
        - lr: learning rate.
    """

    if args.dataset in N_CLASSES_PER_DATASET:
        args.num_classes = N_CLASSES_PER_DATASET[args.dataset]
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        args.num_classes = max(
            1,
            len([entry.name for entry in os.scandir(args.train_data_path) if entry.is_dir]),
        )

    # create backbone-specific arguments
    args.backbone_args = {"cifar": args.dataset in ["cifar10", "cifar100"]}
    if "resnet" not in args.encoder and "convnext" not in args.backbone:
        # dataset related for all transformers
        crop_size = args.crop_size[0]
        args.backbone_args["img_size"] = crop_size
        if "vit" in args.encoder:
            args.backbone_args["patch_size"] = args.patch_size

    with suppress(AttributeError):
        del args.patch_size

    if args.data_format == "dali":
        assert args.dataset in ["imagenet100", "imagenet", "custom"]

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9
    if args.optimizer == "lars":
        args.extra_optimizer_args["momentum"] = 0.9
        args.extra_optimizer_args["exclude_bias_n_norm"] = args.exclude_bias_n_norm

    with suppress(AttributeError):
        del args.exclude_bias_n_norm

    if isinstance(args.devices, int):
        args.devices = [args.devices]
    elif isinstance(args.devices, str):
        args.devices = [int(device) for device in args.devices.split(",") if device]

    # adjust lr according to batch size
    try:
        num_nodes = args.num_nodes or 1
    except AttributeError:
        num_nodes = 1

    scale_factor = args.batch_size * len(args.devices) * num_nodes / 256
    args.lr = args.lr * scale_factor
