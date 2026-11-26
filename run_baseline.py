import argparse
from pathlib import Path
import numpy as np

from src.utils import load_data, download_prerequisites
from src.baseline.preprocess import preprocess_data
from src.visualize import visualize_features, plot_class_counts
from src.baseline.train import train
from src.baseline.test import test


def build_parser():

    parser = argparse.ArgumentParser(description="Movie review classification.")
    parser.add_argument("--dataset", type=Path, help="Path to the dataset directory.")

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()

    download_prerequisites()

    # Load data
    print("Loading the data...")
    train_path = args.dataset / "train-00000-of-00001.parquet"
    test_path = args.dataset / "test-00000-of-00001.parquet"
    train_reviews, train_labels = load_data(train_path)
    test_reviews, test_labels = load_data(test_path)
    print("Data loaded!")

    # Visualize the class counts
    plot_class_counts(train_labels, "train class counts")
    plot_class_counts(test_labels, "test class counts") 

    # Preprocess
    print("Preprocessing the data (This might take a while)...")
    X_train, y_train = preprocess_data(train_reviews, train_labels)
    X_test, y_test = preprocess_data(test_reviews, test_labels)
    print("Data preprocessed!")

    # Visualize the preprocessed data
    print("Visualizing the data...")
    visualize_features(X_train, y_train, title="Train data")
    visualize_features(X_test, y_test, title="Test data")

    # Fit and test a baseline model
    print("Fitting the baseline model...")
    baseline_model = train(X_train, y_train, max_iter=1000)
    print("Testing the baseline model...")
    test(baseline_model, X_test, y_test)


if __name__ == "__main__":
    main()