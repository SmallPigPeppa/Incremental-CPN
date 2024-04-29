import torch
import torch.nn as nn
from torchvision.models import resnet50
import re

# def get_class_count_from_filename(filename):
#     # Extract relevant parts of the filename using regular expressions
#     match = re.search(r'task_(\d+)_pretrain_0samples_cifar100_(\d+)_(\d+)_1993_resnet50\.pt', filename)
#     if not match:
#         raise ValueError("Filename does not match expected format")
#
#     # Extract values from the matched groups
#     task_number = int(match.group(1))
#     initial_classes = int(match.group(2))
#     classes_increment = int(match.group(3))
#
#     # Calculate the total number of classes
#     total_classes = initial_classes + task_number * classes_increment
#     return total_classes
#
#
#
# def get_pretrained_encoder(ckpt_path, cifar=True):
#     state = torch.load(ckpt_path, map_location="cpu")
#     num_classes = get_class_count_from_filename(ckpt_path)
#     new_state = {}
#
#     # Processing keys
#     for k in state.keys():
#         if k in ['convnet.conv1.1.weight', 'convnet.conv1.1.bias', 'convnet.conv1.1.running_mean', 'convnet.conv1.1.running_var']:
#             new_key = k.replace("convnet.conv1.1.", "bn1.")
#             new_state[new_key] = state[k]
#         elif k in ['convnet.conv1.0.weight']:
#             new_key = k.replace("convnet.conv1.0.", "conv1.")
#             new_state[new_key] = state[k]
#         elif "convnet" in k and k not in ['convnet.conv1.1.num_batches_tracked']:
#             new_key = k.replace("convnet.", "")
#             new_state[new_key] = state[k]
#         elif "fc" in k:
#             new_key = k.replace("fc", "fc")
#             new_state[new_key] = state[k]
#
#     model = resnet50(num_classes=num_classes)
#     if cifar:
#         model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
#         model.maxpool = nn.Identity()
#     model.load_state_dict(new_state, strict=True)
#     print(f"Loaded {ckpt_path}")
#     return model


def get_class_count_from_filename(filename):
    # Extract relevant parts of the filename using regular expressions
    match = re.search(r'task_(\d+)_pretrain_0samples_cifar100_(\d+)_(\d+)_1993_resnet50\.pt', filename)
    if not match:
        raise ValueError("Filename does not match expected format")

    # Extract values from the matched groups
    task_number = int(match.group(1))
    initial_classes = int(match.group(2))
    classes_increment = int(match.group(3))

    # Calculate the total number of classes
    total_classes = initial_classes + (task_number+1) * classes_increment
    return total_classes



def get_pretrained_encoder(ckpt_path, cifar=True):
    state = torch.load(ckpt_path, map_location="cpu")
    num_classes = get_class_count_from_filename(ckpt_path)
    new_state = {}

    # Processing keys
    for k in state.keys():
        if k in ['convnet.conv1.1.weight', 'convnet.conv1.1.bias', 'convnet.conv1.1.running_mean', 'convnet.conv1.1.running_var']:
            new_key = k.replace("convnet.conv1.1.", "bn1.")
            new_state[new_key] = state[k]
        elif k in ['convnet.conv1.0.weight']:
            new_key = k.replace("convnet.conv1.0.", "conv1.")
            new_state[new_key] = state[k]
        elif "convnet" in k and k not in ['convnet.conv1.1.num_batches_tracked']:
            new_key = k.replace("convnet.", "")
            new_state[new_key] = state[k]
        elif "fc" in k:
            new_key = k.replace("fc", "fc")
            new_state[new_key] = state[k]

    model = resnet50(num_classes=num_classes)
    if cifar:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.maxpool = nn.Identity()
    model.load_state_dict(new_state, strict=True)
    print(f"Loaded {ckpt_path}")
    return model


if __name__ == "__main__":
    # encoder = get_pretrained_encoder(
    #     ckpt_path='/Users/lwz/Downloads/lwf/task_1_pretrain_0samples_cifar100_50_5_1993_resnet50.pt')
    # encoder = get_pretrained_encoder(
    #     ckpt_path='/Users/lwz/Downloads/lwf/task_9_pretrain_0samples_cifar100_50_5_1993_resnet50.pt')
    # encoder = get_pretrained_encoder(
    #     ckpt_path='/Users/lwz/Downloads/lwf/task_3_pretrain_0samples_cifar100_50_10_1993_resnet50.pt')


    encoder = get_pretrained_encoder(
        ckpt_path='/Users/lwz/Downloads/pycil-log/lwf/task_1_pretrain_0samples_cifar100_0_10_1993_resnet50.pt')
    encoder = get_pretrained_encoder(
        ckpt_path='/Users/lwz/Downloads/pycil-log/lwf/task_9_pretrain_0samples_cifar100_0_10_1993_resnet50.pt')
    encoder = get_pretrained_encoder(
        ckpt_path='/Users/lwz/Downloads/pycil-log/lwf/task_3_pretrain_0samples_cifar100_0_20_1993_resnet50.pt')
