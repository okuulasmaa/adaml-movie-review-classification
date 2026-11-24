from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from src.transformer.model import TransformerClassifier
from src.transformer.validate import validate


def train_transformer(train_loader, valid_loader=None, num_features=300, epochs=5, device="cpu", save_dir=Path("models")):
    
    transformer = TransformerClassifier(num_features).to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = np.inf
    train_losses = []
    val_losses = []

    transformer.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for embeddings, lengths, labels in tqdm(train_loader, desc="Training"):
            embeddings, lengths, labels = embeddings.to(device), lengths.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = transformer(embeddings, lengths, device)
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
            val_loss, val_acc = validate(transformer, valid_loader, device)
            val_losses.append(val_loss)
            print(f"Validation results:    Loss: {val_loss:.3f}, Accuracy: {val_acc:.3f}")
            
            if val_loss < best_val_loss:
                torch.save(transformer.state_dict(), save_dir / "transformer_best.pt")
                best_val_loss = val_loss

        torch.save(transformer.state_dict(), save_dir / "transformer_last.pt")

    return transformer, train_losses, val_losses