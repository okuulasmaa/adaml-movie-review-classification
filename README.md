## Usage
```python3 run_baseline.py --dataset [dataset directory]``` for baseline preprocessing and modeling.
```python3 run_transformer.py --dataset [dataset directory]``` for the transformer training and testing pipeline.

The dataset directory should contain files ```train-00000-of-00001.parquet``` and ```test-00000-of-00001.parquet```

Download the required spaCy pipeline using ```python -m spacy download en_core_web_md``` for the baseline modeling.
