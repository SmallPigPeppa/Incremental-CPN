import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm


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
