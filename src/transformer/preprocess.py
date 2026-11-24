from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import spacy


def preprocess_data(text_array: np.ndarray, labels: np.ndarray, max_tokens=np.inf):

    nlp = spacy.load('en_core_web_md')

    stop_words = set(stopwords.words("english"))

    embeddings_list = []
    lengths = []

    for text in text_array:
        # Text into tokens and filtering out unwanted tokens.
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

        # Limit the number of tokens
        tokens = tokens[:max_tokens] if len(tokens) > max_tokens else tokens

        # Finding the word embeeddings
        embeddings = np.array([nlp.vocab[word].vector for word in tokens])

        embeddings_list.append(embeddings)

    return embeddings_list, labels

