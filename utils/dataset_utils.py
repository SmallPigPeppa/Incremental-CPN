import torchvision
from torchvision import transforms


def get_dataset(dataset, data_path):
    assert dataset in ["cifar100", "imagenet100"]
    if dataset == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        cifar_transforms = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                      transform=cifar_transforms,
                                                      download=True)
        test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                                     transform=cifar_transforms,
                                                     download=True)
    elif dataset == "imagenet100":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return train_dataset, test_dataset
