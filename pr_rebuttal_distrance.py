import torch
from torch.utils.data import DataLoader
from utils.dataset_utils import get_dataset_joint, split_dataset
from utils.encoder_utils_lwf import get_pretrained_encoder
from utils.args_utils import parse_args_cpn
import numpy as np


def compute_mean_vectors(dataloader, model, device):
    class_means = {}
    class_counts = {}
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            features = model(data)
            for i, label in enumerate(target):
                if label.item() not in class_means:
                    class_means[label.item()] = features[i]
                    class_counts[label.item()] = 1
                else:
                    class_means[label.item()] += features[i]
                    class_counts[label.item()] += 1
    for label in class_means:
        class_means[label] = class_means[label] / class_counts[label]
    return class_means


def main():
    args = parse_args_cpn()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classes_order = torch.tensor(
        [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28,
         53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97,
         2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7,
         63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33])
    tasks = classes_order.chunk(args.num_tasks)
    train_dataset, test_dataset = get_dataset_joint(dataset=args.dataset, data_path=args.data_path)

    # Only use the dataset of task 0 for all tests
    test_dataset_task0 = split_dataset(test_dataset, tasks=tasks, task_idx=[0])
    test_loader_task0 = DataLoader(test_dataset_task0, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                   pin_memory=True)

    initial_class_means = None
    displacements = []

    for task_idx in range(args.num_tasks):
        filename = f"{args.pretrained_model}/task_{task_idx}_pretrain_0samples_cifar100_0_{100 // args.num_tasks}_1993_resnet50.pt"
        encoder = get_pretrained_encoder(filename, cifar=("cifar" in args.dataset))
        encoder.fc = torch.nn.Identity()
        encoder = encoder.to(device)

        if task_idx == 0:
            initial_class_means = compute_mean_vectors(test_loader_task0, encoder, device)

        current_class_means = compute_mean_vectors(test_loader_task0, encoder, device)
        displacement = [torch.norm(current_class_means[label] - initial_class_means[label]).item() for label in
                        initial_class_means.keys()]
        mean_displacement = np.mean(displacement)
        displacements.append(mean_displacement)

    print("Average displacements per task:", displacements)


if __name__ == '__main__':
    main()
