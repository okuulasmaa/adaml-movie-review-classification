from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from src.transformer.model import TransformerClassifier
from src.transformer.validate import validate
from src.utils import create_padding_mask


def train_transformer(train_loader, valid_loader=None, num_features=300, nhead=6, vocab_size=5000, num_classes=2, epochs=5, class_weights=None, device="cpu", save_dir=Path("models")):
    
    transformer = TransformerClassifier(
        num_features=num_features, 
        num_classes=num_classes,
        vocab_size=vocab_size,
        nhead=nhead
    ).to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = np.inf
    train_losses = []
    val_losses = []

    transformer.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for tokens, lengths, labels in tqdm(train_loader, desc="Training"):
            tokens, lengths, labels = tokens.to(device), lengths.to(device), labels.to(device)

            optimizer.zero_grad()
            mask = create_padding_mask(lengths, tokens.shape[1], device)
            outputs = transformer(tokens, mask)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()

        train_losses.append(total_loss)
        total = len(train_loader.dataset)
        accuracy = correct / total

        print(f"Epoch {epoch+1}:    Loss: {total_loss:.3f}, Accuracy: {accuracy:.3f}")

        if not valid_loader is None:
            val_loss, val_acc, f1 = validate(transformer, valid_loader, device)
            val_losses.append(val_loss)
            print(f"Validation results:    Loss: {val_loss:.3f}, Accuracy: {val_acc:.3f}, F1: {f1:.3f}")
            
            if val_loss < best_val_loss:
                torch.save(transformer.state_dict(), save_dir / "transformer_best.pt")
                best_val_loss = val_loss

        torch.save(transformer.state_dict(), save_dir / "transformer_last.pt")

    return transformer, train_losses, val_losses