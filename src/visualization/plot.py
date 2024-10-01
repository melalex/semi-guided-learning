from matplotlib import pyplot as plt


def plot_loss_and_val_loss(loss: list[float], val_loss: list[float], size=(12, 6)) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_loss_and_val_accuracy(
    accuracy: list[float], val_accuracy: list[float], size=(12, 6)
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy, label="Training Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
