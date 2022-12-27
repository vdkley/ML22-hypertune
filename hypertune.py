from pathlib import Path
from typing import Dict

import ray
import torch
from torch.utils.data import DataLoader
from filelock import FileLock

# from loguru import logger
from ray import tune
from ray.tune import CLIReporter

# from ray.tune import JupyterNotebookReporter
# from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

from src.data import data_tools, make_dataset
from src.models import metrics, tokenizer, rnn_models, train_model
from src.settings import SearchSpace


def train(config: Dict, checkpoint_dir: str = None) -> None:
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """

    data_dir = config['data_dir']
    with FileLock(data_dir / ".lock"):
        trainpaths, testpaths = make_dataset.get_imdb_data(data_dir)
    traindataset = data_tools.TextDataset(paths=trainpaths)
    testdataset = data_tools.TextDataset(paths=testpaths)

    corpus = []
    for i in range(len(traindataset)):
        x = tokenizer.clean(traindataset[i][0])
        corpus.append(x)
    v = tokenizer.build_vocab(corpus, max=10000)
    len(v)

    preprocessor = tokenizer.Preprocessor(max=100, vocab=v, clean=tokenizer.clean)
    trainloader = DataLoader(
        traindataset, collate_fn=preprocessor, batch_size=32, shuffle=True
    )
    testloader = DataLoader(
        testdataset, collate_fn=preprocessor, batch_size=32, shuffle=True
    )

    accuracy = metrics.Accuracy()
    loss_fn = torch.nn.CrossEntropyLoss()
    log_dir = Path("models/attention/")
    model = rnn_models.AttentionNLP(config)
    model = train_model.trainloop(
        epochs=10,
        model=model,
        metrics=[accuracy],
        optimizer=torch.optim.Adam,
        learning_rate=1e-3,
        loss_fn=loss_fn,
        train_dataloader=trainloader,
        test_dataloader=testloader,
        log_dir=log_dir,
        train_steps=100,
        eval_steps=25,
        tunewriter=True,
    )


if __name__ == "__main__":
    ray.init()

    # have a look in src.settings to see how SearchSpace is created.
    # If you want to search other ranges, you change this in the settings file.
    # for AttentionNLP model config:  vocab=input_size, hidden_size, dropout, num_layers, output_size
    config = SearchSpace(
        input_size=10002,
        output_size=2,
        tune_dir=Path("models/ray").resolve(),
        data_dir=Path("data/raw").resolve(),
    )

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()

    # zie documentatie:
    # https://docs.ray.io/en/releases-1.12.1/tune/api_docs/execution.html
    analysis = tune.run(
        train,
        config=config.dict(),
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        local_dir=config.tune_dir,
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()
