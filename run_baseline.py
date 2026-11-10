from pathlib import Path

from src.baseline.utils import load_data, download_prerequisites
from src.baseline.preprocess import preprocess_data
from src.visualize import visualize_features
from src.baseline.train import train
from src.baseline.test import test


def main():

    download_prerequisites()

    # Load data
    print("Loading the data...")
    train_path = Path("data/train-00000-of-00001.parquet")
    test_path = Path("data/test-00000-of-00001.parquet")
    train_reviews, train_labels = load_data(train_path)
    test_reviews, test_labels = load_data(test_path)
    print("Data loaded!")

    # Preprocess
    print("Preprocessing the data (This might take a while)...")
    X_train, y_train = preprocess_data(train_reviews, train_labels)
    X_test, y_test = preprocess_data(test_reviews, test_labels, max_length=X_train.shape[1])
    print("Data preprocessed!")

    # Visualize the preprocessed data
    print("Visualizing the data...")
    visualize_features(X_train, y_train, title="Train data")
    visualize_features(X_test, y_test, title="Test data")

    # Fit and test a baseline model
    baseline_model = train(X_train, y_train)
    test(baseline_model, X_test, y_test)


if __name__ == "__main__":
    main()