from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

# import gin
# import numpy as np
# import pandas as pd
# import requests
import tensorflow as tf
import torch
from loguru import logger

from src.data import data_tools

# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

# from src.data.data_tools import PaddedDatagenerator, TSDataset

Tensor = torch.Tensor


def get_imdb_data(cache_dir: str = ".") -> Tuple[List[Path], List[Path]]:
    datapath = Path(cache_dir) / "aclImdb"
    if datapath.exists():
        logger.info(f"{datapath} already exists, skipping download")
    else:
        logger.info(f"{datapath} not found on disk, downloading")

        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

        _ = tf.keras.utils.get_file(
            "aclImdb_v1.tar.gz", url, untar=True, cache_dir=cache_dir, cache_subdir=""
        )
    testdir = datapath / "test"
    traindir = datapath / "train"
    keep_subdirs_only(testdir)
    keep_subdirs_only(traindir)
    unsup = traindir / "unsup"
    if unsup.exists():
        shutil.rmtree(traindir / "unsup")
    formats = [".txt"]
    testpaths = [
        path for path in data_tools.walk_dir(testdir) if path.suffix in formats
    ]
    trainpaths = [
        path for path in data_tools.walk_dir(traindir) if path.suffix in formats
    ]
    return trainpaths, testpaths


def keep_subdirs_only(path: Path) -> None:
    files = [file for file in path.iterdir() if file.is_file()]
    for file in files:
        file.unlink()
