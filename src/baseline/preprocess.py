from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import spacy


def preprocess_data(text_list: list[str], labels_list: list[int], max_length: int=0):
    '''
    Tokenizes the texts and converts the tokens into integers using hash()
    '''

    nlp = spacy.load('en_core_web_md')

    stop_words = set(stopwords.words("english"))

    data_rows = []

    for i, text in enumerate(text_list):
        # Text into tokens and filtering out unwanted tokens.
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        
        # Finding the word embeeddings
        embeddings = np.array([nlp.vocab[word].vector for word in tokens])

        # Mean aggregate the embeddigns
        embeddings = embeddings.sum(axis=0) / embeddings.shape[0]
        data_rows.append(embeddings)

    X = np.array(data_rows)
    y = np.array(labels_list)

    return X, y
