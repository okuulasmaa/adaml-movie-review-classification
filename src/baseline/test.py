
import numpy as np
from sklearn.linear_model import LogisticRegression


def test(model: LogisticRegression, X, y):

    preds = model.predict(X)
    correct = np.sum(y == preds)

    print(f"Baseline accuracy: {(correct / len(preds) * 100):.3f} %")
