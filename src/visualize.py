from umap import UMAP
import matplotlib.pyplot as plt


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
    path = "figs/"+title+".pdf"
    plt.savefig(path)
    plt.show()