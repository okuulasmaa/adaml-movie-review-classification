from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader

from src.utils import load_data, download_prerequisites, MovieReviewDataset, collate_fn
from src.transformer.preprocess import preprocess_data
from src.visualize import visualize_features, plot_loss_curve
from src.transformer.model import TransformerClassifier
from src.transformer.train import train_transformer
from src.transformer.validate import validate



def build_parser():

    parser = argparse.ArgumentParser(description="Movie review classification.")
    parser.add_argument("--skip_train", action="store_true", help="Skip the training and only evaluate a saved model")

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()

    download_prerequisites()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading the data
    train_path = Path("data/train-00000-of-00001.parquet")
    test_path = Path("data/test-00000-of-00001.parquet")
    train_reviews, train_labels = load_data(train_path)
    test_reviews, test_labels = load_data(test_path)
    print("Data loaded!")

    # Preprocessing including finding the pre-calculated embeddings
    print("Preprocessing the data (This might take a while...)")
    # Set max_tokens to e.g. 200 or 150 if you run out of memory during training.
    X_train, y_train = preprocess_data(train_reviews, train_labels)
    X_test, y_test = preprocess_data(test_reviews, test_labels)
    print("Data preprocessed!")

    # Datasets
    batch_size = 16
    train_dataset = MovieReviewDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_dataset = MovieReviewDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # Training
    if not args.skip_train:
        save_dir = Path("models")
        transformer_model, train_losses, val_losses = train_transformer(train_loader, valid_loader=test_loader, epochs=15, device=device, save_dir=save_dir)

        # Training visualization
        plot_loss_curve(train_losses, val_losses)

    # Testing
    transformer_model = TransformerClassifier().to(device)
    transformer_model.load_state_dict(torch.load(save_dir / "transformer_best.pt", weights_only=True))
    val_loss, val_acc = validate(transformer_model, test_loader, device)
    print(f"Validation results:    Loss: {val_loss:.3f}, Accuracy: {val_acc:.3f}")


if __name__ == "__main__":
    main()