import torch
import torch.nn as nn
from tqdm import tqdm


def validate(transformer, valid_loader, device="cpu"):
    transformer.eval()

    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for embeddings, lengths, labels in tqdm(valid_loader, desc="Validating"):
            embeddings, lengths, labels = embeddings.to(device), lengths.to(device), labels.to(device)

            outputs = transformer(embeddings, lengths, device)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()

    total = len(valid_loader.dataset)
    accuracy = correct / total

    return total_loss, accuracy