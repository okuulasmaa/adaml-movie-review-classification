from pathlib import Path
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def visualize_features(X, y, title=""):

    umap = UMAP(n_neighbors=15, n_components=2)
    X_umap = umap.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], s=1, c=y)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    fig.suptitle(title)
    fig.legend(*scatter.legend_elements(), loc="outside center right")
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    path = Path("figs/"+title+".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.show()


def plot_confusion_matrix(preds, labels, classes):

    cm = confusion_matrix(labels, preds, labels=classes)
    cm = cm.astype(np.float64) / np.sum(cm, axis=1) # Normalization
    cm_disp = ConfusionMatrixDisplay(cm, display_labels=classes)

    cm_disp.plot(cmap='gray')

    plt.show()


def plot_loss_curve(train_losses, val_losses):

    epochs = range(1, len(train_losses) + 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(epochs, train_losses)
    ax.plot(epochs, val_losses)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(["Training", "Validation"])

    path = Path("figs/loss_curve.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.show()


def plot_class_counts(labels: np.ndarray, title="Class counts"):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Find label counts
    labels = labels.tolist()
    label_counts = {label: labels.count(label) for label in np.unique(labels)}

    ax.bar(label_counts.keys(), label_counts.values())

    ax.set_title(title)

    path = Path(f"figs/{title.replace(" ", "_")}.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.show()