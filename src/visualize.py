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