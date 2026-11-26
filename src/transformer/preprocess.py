from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

from src.utils import Vocabulary


def preprocess_data(text_array: np.ndarray, labels: np.ndarray, max_tokens=np.inf, vocab=Vocabulary()):

    stop_words = set(stopwords.words("english"))

    encoding_list = []
    remove_idx = []

    for i, text in enumerate(text_array):
        # Text into tokens and filtering out unwanted tokens.
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

        # Limit the number of tokens
        tokens = tokens[:max_tokens] if len(tokens) > max_tokens else tokens

        if len(tokens) < 1:
            remove_idx.append(i)
            continue

        encodings = []
        for token in tokens:
            vocab.insert(token)
            encodings.append(vocab.get_encoding(token))
        encoding_list.append(encodings)

    labels = np.delete(labels, remove_idx)

    return encoding_list, labels, vocab



