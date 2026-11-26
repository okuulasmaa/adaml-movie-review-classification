import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
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
SEED = 42

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

    torch.manual_seed(SEED)

    model_dir = Path("models") / args.dataset.name

    # Loading the data
    train_path = args.dataset / "train-00000-of-00001.parquet"
    test_path = args.dataset / "test-00000-of-00001.parquet"
    train_reviews, train_labels = load_data(train_path)
    test_reviews, test_labels = load_data(test_path)
    print("Data loaded!")

    print("Preprocessing the data (This might take a while...)")
    X_train, y_train, vocab = preprocess_data(train_reviews, train_labels, max_tokens=200)
    X_test, y_test, _ = preprocess_data(test_reviews, test_labels, max_tokens=200, vocab=vocab)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, shuffle=True, test_size=0.5, random_state=SEED)
    class_weights = torch.tensor(compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)).to(device=device, dtype=torch.float)
    num_classes = len(np.unique(y_test))
    print("Data preprocessed!")
    
    # Datasets
    train_dataset = TokenDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_dataset = TokenDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    test_dataset = TokenDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    # Visualize the class counts
    plot_class_counts(train_labels, "train class counts")
    plot_class_counts(test_labels, "test and validation class counts") 

    # Training
    if not args.skip_train:
        print(f"Training for {EPOCHS} epochs...")
        transformer_model, train_losses, val_losses = train_transformer(
            train_loader=train_loader, 
            valid_loader=val_loader, 
            num_features=NUM_FEATURES, 
            nhead=NHEAD, 
            vocab_size=vocab.word_count, 
            num_classes=num_classes, 
            epochs=10, 
            class_weights=class_weights,
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

    test_loss, test_acc, f1 = validate(transformer_model, test_loader, device, verbose=True)
    print(f"Testing results:    Loss: {test_loss:.3f}, Accuracy: {test_acc:.3f}, F1: {f1:.3f}")


if __name__ == "__main__":
    main()