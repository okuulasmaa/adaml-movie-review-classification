from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import spacy


def preprocess_data(text_array: np.ndarray, labels: np.ndarray):
    
    nlp = spacy.load('en_core_web_md')

    stop_words = set(stopwords.words("english"))

    data_rows = []
    remove_idx = []

    for i, text in enumerate(text_array):
        # Text into tokens and filtering out unwanted tokens.
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

        # Finding the word embeeddings
        embeddings = np.array([nlp.vocab[word].vector for word in tokens])

        if len(tokens) < 1:
            remove_idx.append(i)
            continue

        # Mean aggregate the embeddigns
        embeddings = embeddings.sum(axis=0) / embeddings.shape[0]
        data_rows.append(embeddings)
        
    X = np.array(data_rows)
    y = np.delete(labels, remove_idx)

    return X, y
