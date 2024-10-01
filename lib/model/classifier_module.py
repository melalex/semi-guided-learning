from dataclasses import dataclass
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import tqdm.notebook

from lib.util.progress_bar import NOTEBOOK_PROGRESS_BAR, ProgressBarProvider


@dataclass
class TrainFeedback:
    train_accuracy_history: list[float]
    train_loss_history: list[float]
    valid_accuracy_history: list[float]
    valid_loss_history: list[float]


class ClassifierModule(nn.Module):

    def fit(
        self,
        num_epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        loss_fun: nn.Module,
        optimizer: Optimizer,
        progress_bar: ProgressBarProvider = NOTEBOOK_PROGRESS_BAR,
    ) -> TrainFeedback:
        num_batches = len(train_loader)
        train_accuracy_history = []
        train_loss_history = []
        valid_accuracy_history = []
        valid_loss_history = []

        for epoch in progress_bar.provide(num_epochs, desc="Overall progress"):
            with progress_bar.provide(total=num_batches) as p_bar:
                p_bar.set_description("Epoch [%s]" % (epoch + 1))

                train_total_loss = 0
                train_total_accuracy = 0

                for _, (x, y_true) in enumerate(train_loader):
                    x = x.to(self.device)
                    y_true = y_true.to(self.device)

                    outputs = self(x)

                    accuracy = self.__calculate_accuracy(outputs.data, y_true)

                    train_total_accuracy += accuracy

                    loss = loss_fun(outputs, y_true)

                    train_total_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    p_bar.update()
                    p_bar.set_postfix(loss=loss.item(), accuracy=accuracy)

                train_avg_loss = train_total_loss / num_batches
                train_avg_accuracy = train_total_accuracy / num_batches

                train_loss_history.append(train_avg_loss)
                train_accuracy_history.append(train_avg_accuracy)

                valid_loss, valid_accuracy = self.test(loss_fun, valid_loader)

                valid_accuracy_history.append(valid_accuracy)
                valid_loss_history.append(valid_loss)

                p_bar.set_postfix(
                    loss=train_avg_loss,
                    accuracy=train_avg_accuracy,
                    valid_loss=valid_loss,
                    valid_accuracy=valid_accuracy,
                )

        return TrainFeedback(
            train_accuracy_history,
            train_loss_history,
            valid_accuracy_history,
            valid_loss_history,
        )

    def test(self, loss_fun: nn.Module, loader: DataLoader) -> tuple[float, float]:
        with torch.no_grad():
            total_loss = 0
            total_accuracy = 0
            batch_count = len(loader)

            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                total_loss += loss_fun(outputs, labels).item()
                total_accuracy += self.__calculate_accuracy(outputs.data, labels)

            return (total_loss / batch_count, total_accuracy / batch_count)

    def __calculate_accuracy(self, y_predicted: torch.Tensor, y_true: torch.Tensor):
        _, predicted = torch.max(y_predicted, 1)
        return (predicted == y_true).sum().item() / y_true.size(0)
