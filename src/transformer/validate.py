import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.visualize import plot_confusion_matrix
from src.utils import create_padding_mask


def validate(transformer, valid_loader, device="cpu", verbose=False):
    transformer.eval()

    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for tokens, lengths, labels in tqdm(valid_loader, desc="Validating"):
            tokens, lengths, labels = tokens.to(device), lengths.to(device), labels.to(device)

            mask = create_padding_mask(lengths, tokens.shape[1], device)
            outputs = transformer(tokens, mask)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu())
            all_preds.extend(preds.cpu()) 

    total = len(valid_loader.dataset)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')

    if verbose:
        plot_confusion_matrix(all_preds, all_labels, np.unique(all_labels))

    return total_loss, accuracy, f1