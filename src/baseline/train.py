from sklearn.linear_model import LogisticRegression


def train(X, y, max_iter=100):

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X, y)

    return model

