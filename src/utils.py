from pathlib import Path
import pandas as pd
import nltk
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Vocabulary:

    def __init__(self):
        self.word_count = 1
        self.vocabulary = {}

    def insert(self, word):
        if not word in self.vocabulary.keys():
            self.vocabulary[word] = self.word_count
            self.word_count += 1

    def get_encoding(self, word):
        try:
            return self.vocabulary[word]
        except KeyError:
            return -1


def load_data(data_path: Path) -> tuple[list, list]:

    data_df = pd.read_parquet(data_path)
    reviews = data_df["text"].to_numpy()
    labels = data_df["label"].to_numpy()

    return reviews, labels


def download_prerequisites():

    nltk.download("stopwords")
    nltk.download('punkt_tab')


# By Copilot
class TokenDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(
            self.tokens[idx],
            dtype=torch.long
        )

        y = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return x, y


# By Copilot
def collate_fn(batch):
    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [seq.size(0) for seq in sequences]
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0
    )

    labels = torch.tensor(labels)

    return padded_sequences, lengths, labels


# By Copilot
def create_padding_mask(lengths, max_len, device):
    return (
        torch.arange(max_len, device=device)
        .expand(len(lengths), max_len)
        >= lengths.unsqueeze(1)
    )