import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.utils import load_data, download_prerequisites, TokenDataset, collate_fn
from src.transformer.preprocess import preprocess_data
from src.visualize import visualize_features, plot_loss_curve, plot_class_counts
from src.transformer.model import TransformerClassifier
from src.transformer.train import train_transformer
from src.transformer.validate import validate

NUM_FEATURES = 400
NHEAD = 8
EPOCHS = 10
BATCH_SIZE = 16

def build_parser():

    parser = argparse.ArgumentParser(description="Movie review classification.")
    parser.add_argument("--dataset", type=Path, help="Path to the dataset directory.")
    parser.add_argument("--skip_train", action="store_true", help="Skip the training and only evaluate a saved model")

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()

    download_prerequisites()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    model_dir = Path("models")

    # Loading the data
    train_path = args.dataset / "train-00000-of-00001.parquet"
    test_path = args.dataset / "test-00000-of-00001.parquet"
    train_reviews, train_labels = load_data(train_path)
    test_reviews, test_labels = load_data(test_path)
    print("Data loaded!")

    print("Preprocessing the data (This might take a while...)")
    X_train, y_train, vocab = preprocess_data(train_reviews, train_labels, max_tokens=200)
    X_test, y_test, _ = preprocess_data(test_reviews, test_labels, max_tokens=200, vocab=vocab)
    num_classes = len(np.unique(y_test))
    print("Data preprocessed!")
    
    # Datasets
    train_dataset = TokenDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    test_dataset = TokenDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    # Visualize the class counts
    plot_class_counts(train_labels, "train class counts")
    plot_class_counts(test_labels, "test class counts") 

    # Training
    if not args.skip_train:
        print(f"Training for {EPOCHS} epochs...")
        transformer_model, train_losses, val_losses = train_transformer(
            train_loader=train_loader, 
            valid_loader=test_loader, 
            num_features=NUM_FEATURES, 
            nhead=NHEAD, 
            vocab_size=vocab.word_count, 
            num_classes=num_classes, 
            epochs=10, 
            device=device, 
            save_dir=model_dir
        )
        print("Training completed!")

        # Training visualization
        plot_loss_curve(train_losses, val_losses)

    # Testing
    transformer_model = TransformerClassifier(
        num_features=NUM_FEATURES, 
        num_classes=num_classes,
        vocab_size=vocab.word_count,
        nhead=NHEAD
    ).to(device)
    transformer_model.load_state_dict(torch.load(model_dir / "transformer_best.pt", weights_only=True))

    val_loss, val_acc, f1 = validate(transformer_model, test_loader, device, verbose=True)
    print(f"Validation results:    Loss: {val_loss:.3f}, Accuracy: {val_acc:.3f}, F1: {f1:.3f}")


if __name__ == "__main__":
    main()