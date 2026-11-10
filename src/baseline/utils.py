from pathlib import Path
import pandas as pd
import nltk

def load_data(data_path: Path) -> tuple[list, list]:

    data_df = pd.read_parquet(data_path)
    reviews = data_df["text"].to_list()
    labels = data_df["label"].to_list()

    return reviews, labels


def download_prerequisites():

    nltk.download("stopwords")
    nltk.download('punkt_tab')
