import matplotlib.pyplot as plt


def plot_gradients(grad_magnitudes, title="Gradient Magnitude Over Iterations"):
    plt.figure(figsize=(10, 6))
    for key, values in grad_magnitudes.items():
        plt.plot(values, label=f'Gradient Magnitude {key}')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Magnitude')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy(accuracies, val_accuracies=None, title="Accuracy Over Epochs"):
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, label='Accuracy')
    if val_accuracies:
        plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
