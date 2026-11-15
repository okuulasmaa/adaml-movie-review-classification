from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import spacy


def preprocess_data(text_array: np.ndarray, labels: np.ndarray):
    '''
    Tokenizes the texts and converts the tokens into integers using hash()
    '''

    nlp = spacy.load('en_core_web_md')

    stop_words = set(stopwords.words("english"))

    data_rows = []

    for text in text_array:
        # Text into tokens and filtering out unwanted tokens.
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

        # Finding the word embeeddings
        embeddings = np.array([nlp.vocab[word].vector for word in tokens])

        # Mean aggregate the embeddigns
        embeddings = embeddings.sum(axis=0) / embeddings.shape[0]
        data_rows.append(embeddings)
        
    X = np.array(data_rows)
    y = labels if type(labels) is np.ndarray else np.array(labels)

    return X, y
