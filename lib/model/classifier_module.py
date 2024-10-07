import copy
import math
from pathlib import Path
import numpy as np
import torch

from dataclasses import dataclass
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

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
        initial_patience: int = -1,
        progress_bar: ProgressBarProvider = NOTEBOOK_PROGRESS_BAR,
    ) -> TrainFeedback:
        num_batches = len(train_loader)
        train_accuracy_history = []
        train_loss_history = []
        valid_accuracy_history = []
        valid_loss_history = []
        best_accuracy = 0
        best_model_weights = None
        patience = initial_patience

        for epoch in progress_bar.provide(num_epochs, desc="Overall progress"):
            with progress_bar.provide(total=num_batches) as p_bar:
                p_bar.set_description("Epoch [%s]" % (epoch + 1))

                train_total_loss = 0
                train_total_accuracy = 0

                for x, y_true in train_loader:
                    x = x.to(self.device)
                    y_true = y_true.to(self.device)

                    outputs = self(x)

                    accuracy = self.__calculate_accuracy(
                        self.__extract_prediction(outputs.data), y_true
                    )

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

                valid_loss, valid_accuracy, _ = self.test(loss_fun, valid_loader)

                valid_accuracy_history.append(valid_accuracy)
                valid_loss_history.append(valid_loss)

                # Early stopping
                if patience != -1:
                    if valid_accuracy > best_accuracy:
                        best_accuracy = valid_accuracy
                        best_model_weights = copy.deepcopy(self.state_dict())
                        patience = initial_patience
                    else:
                        patience -= 1

                p_bar.set_postfix(
                    loss=train_avg_loss,
                    accuracy=train_avg_accuracy,
                    valid_loss=valid_loss,
                    valid_accuracy=valid_accuracy,
                    patience=patience,
                )

                if patience == 0:
                    break

        if best_model_weights is not None:
            self.load_state_dict(best_model_weights)

        return TrainFeedback(
            train_accuracy_history,
            train_loss_history,
            valid_accuracy_history,
            valid_loss_history,
        )

    def test(
        self, loss_fun: nn.Module, loader: DataLoader, record_class_stats: bool = False
    ) -> tuple[float, float, np.array]:
        with torch.no_grad():
            total_loss = 0
            total_accuracy = 0
            batch_count = len(loader)
            confusion_matrix = None

            if record_class_stats:
                confusion_matrix = np.zeros(
                    (self.num_classes, self.num_classes), dtype=np.uint8
                )

            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                total_loss += loss_fun(outputs, labels).item()
                y_predicted = self.__extract_prediction(outputs.data)
                total_accuracy += self.__calculate_accuracy(y_predicted, labels)

                if record_class_stats:
                    for t, p in zip(labels.view(-1), y_predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

            return (
                total_loss / batch_count,
                total_accuracy / batch_count,
                confusion_matrix,
            )

    def forward_loader(self, loader: DataLoader) -> Tensor:

        def forward_batch(img: Tensor) -> Tensor:
            outputs = self(img.to(self.device))
            return self.__extract_prediction(outputs.data).cpu()

        return torch.cat([forward_batch(images) for images, _ in loader])

    def persist(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), path)

    def __calculate_accuracy(self, y_predicted: torch.Tensor, y_true: torch.Tensor):
        return (y_predicted == y_true).sum().item() / y_true.size(0)

    def __extract_prediction(self, y_predicted: torch.Tensor):
        _, predicted = torch.max(y_predicted, 1)
        return predicted
