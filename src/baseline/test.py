
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.visualize import plot_confusion_matrix


def test(model: LogisticRegression, X, y):

    preds = model.predict(X)

    # Accuracy
    correct = np.sum(y == preds)
    accuracy = correct / len(preds)
    print(f"Baseline accuracy: {(accuracy * 100):.3f} %")

    # F1
    f1 = f1_score(y, preds)
    print(f"Baseline f1: {f1:.4f}")

    plot_confusion_matrix(preds, y, np.unique(y))
