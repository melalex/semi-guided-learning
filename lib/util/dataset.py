from torch.utils.data import Dataset
from torch import Tensor, nn


class SemiSupervisedDataset(Dataset):
    __original_dataset: Dataset
    __labels: Tensor

    def __init__(self, original_dataset: Dataset, labels: Tensor):
        self.__original_dataset = original_dataset
        self.__labels = labels

    def __len__(self):
        return len(self.__original_dataset)

    def __getitem__(self, idx):
        sample, _ = self.__original_dataset[idx]
        label = self.__labels[idx]

        return sample, label


class TransformerDataset(Dataset):
    decorated: Dataset
    transform: nn.Module

    def __init__(self, decorated: Dataset, transform: nn.Module):
        self.decorated = decorated
        self.transform = transform

    def __len__(self):
        return len(self.decorated)

    def __getitem__(self, idx):
        image, target = self.decorated[idx]

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


class WarmCachedDataset(Dataset):
    __cache: list[tuple[any, any]]

    def __init__(self, decorated) -> None:
        self.__cache = []
        for value, label in decorated:
            self.__cache.append((value, label))

    def __len__(self):
        return len(self.__cache)

    def __getitem__(self, idx):
        return self.__cache[idx]
