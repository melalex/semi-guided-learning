from matplotlib import pyplot as plt
import numpy as np
from torchvision.datasets.vision import VisionDataset
import torch


def sample_image_dataset(
    dataset: VisionDataset,
    labels_map: dict[int, str] = {},
    mean=0,
    std=1,
    cols: int = 3,
    rows: int = 3,
):
    figure = plt.figure(figsize=(8, 8))

    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        un_normalized_img = img * std + mean
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label] if label in labels_map else label)
        plt.axis("off")
        plt.imshow(np.transpose(un_normalized_img, (1, 2, 0)))

    plt.show()
