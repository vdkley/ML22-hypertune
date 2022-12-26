import sys

sys.path.insert(0, "../..")
from src.data import data_tools, make_dataset
from torch.utils.data import DataLoader
from src.models import tokenizer, train_model
import torch
from src.models import metrics
from pathlib import Path


data_dir = "data/raw"
trainpaths, testpaths = make_dataset.get_imdb_data(data_dir)
traindataset = data_tools.TextDataset(paths=trainpaths)
testdataset = data_tools.TextDataset(paths=testpaths)

