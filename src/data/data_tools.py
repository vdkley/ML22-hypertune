# from __future__ import annotations

# import random
# import shutil
# from datetime import datetime
from pathlib import Path
from typing import Iterator

# import numpy as np
# import tensorflow as tf
# import torch
# import requests
# import zipfile
# from loguru import logger
# from torch.nn.utils.rnn import pad_sequence
# from tqdm import tqdm

# Tensor = torch.Tensor


def walk_dir(path: Path) -> Iterator:
    """loops recursively through a folder

    Args:
        path (Path): folder to loop trough. If a directory
            is encountered, loop through that recursively.

    Yields:
        Generator: all paths in a folder and subdirs.
    """

    for p in Path(path).iterdir():
        if p.is_dir():
            yield from walk_dir(p)
            continue
        # resolve works like .absolute(), but it removes the "../.." parts
        # of the location, so it is cleaner
        yield p.resolve()
