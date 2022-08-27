import numpy as np
import torch.nn as nn
import torch
import tqdm

IMGSIZE = 32
LR = 0.1
GPUS = [0]
BS0 = 128
BS2 = 512
# ckpt_dir='/mnt/mmtech01/usr/liuwenzhuo/code/solo-learn/trained_models/simclr/2mv95572'
# ckpt_dir='/share/wenzhuoliu/code/solo-learn/trained_models/byol/t3pmk238'
# ckpt_dir='/share/wenzhuoliu/code/solo-learn/trained_models/mocov2plus/1kguyx5e'
ckpt_dir = '/share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/1ehqqmug'

# ckpt_dir='/share/wenzhuoliu/code/'
data_path = '/share/wenzhuoliu/torch_ds'

# for filename in os.listdir(ckpt_dir):
#     basename, ext = os.path.splitext(filename)
#     if ext == '.ckpt':
#         ckpt_path = os.path.join(ckpt_dir, filename)
#         print(f'load ckpt from {ckpt_path}')

ckpt_path = '/share/wenzhuoliu/code/solo-learn/trained_models/swav/yaaves5o/swav-imagenet32-yaaves5o-ep=999.ckpt'
# ckpt_path = '/share/wenzhuoliu/code/solo-learn/trained_models/barlow_twins/s5fh5bvf/barlow_twins-imagenet32-s5fh5bvf-ep=999.ckpt'

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
encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
encoder.maxpool = nn.Identity()
encoder.fc = nn.Identity()
encoder.load_state_dict(state, strict=False)
print(f"Loaded {ckpt_path}")
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
encoder.eval()
encoder.to(device)
# cifar100
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]
# # imagenet
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
cifar_transforms = transforms.Compose(
    [transforms.Resize(IMGSIZE), transforms.ToTensor(), transforms.Normalize(mean, std)])
# transforms.CenterCrop(size=96)
train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                              transform=cifar_transforms,
                                              download=True)
test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                             transform=cifar_transforms,
                                             download=True)
# train_dataset = torchvision.datasets.CIFAR100(root='~/torch_ds', split='train',
#                                                            transform=stl_transform,
#                                                            download=True)
# test_dataset = torchvision.datasets.CIFAR100(root='~/torch_ds', split='test',
#                                                            transform=stl_transform,
#                                                            download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8,
                          pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=8,
                         pin_memory=True)
x_train = []
x_test = []
y_train = []
y_test = []
encoder = nn.DataParallel(encoder)
for x, y in tqdm(iter(train_loader)):
    x = x.to(device)
    z = encoder(x)
    x_train.append(z.cpu().detach().numpy())
    y_train.append(y.cpu().detach().numpy())
for x, y in tqdm(iter(test_loader)):
    x = x.to(device)
    z = encoder(x)
    x_test.append(z.cpu().detach().numpy())
    y_test.append(y.cpu().detach().numpy())

x_train = np.vstack(x_train)


def get_means(encoder, train_loader, classes):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    encoder.eval()
    encoder.to(device)
    x_train = []
    y_train = []
    encoder = nn.DataParallel(encoder)
    for x, y in tqdm(iter(train_loader)):
        x = x.to(device)
        z = encoder(x)
        x_train.append(z.cpu().detach().numpy())
        y_train.append(y.cpu().detach().numpy())

    x_train = np.vstack(x_train)
    y_train = np.hstack(y_train)

    means = []
    for i in classes:
        index_i = y_train == i
        x_train_i = x_train[index_i]
        mean_i = np.mean(x_train_i, axis=0)
        means.append(mean_i)
    # return np.array(means)
    return torch.tensor(means)
