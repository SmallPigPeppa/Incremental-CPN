import torch
import torch.nn as nn
import warnings
from torchvision.models import resnet50


def get_pretrained_encoder(ckpt_path, cifar=True):
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
    encoder = resnet50()
    if cifar:
        encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        encoder.maxpool = nn.Identity()
        encoder.fc = nn.Identity()
    encoder.load_state_dict(state, strict=False)
    print(f"Loaded {ckpt_path}")
    return encoder
