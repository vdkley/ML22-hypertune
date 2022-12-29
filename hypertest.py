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



print(len(trainloader))

accuracy = metrics.Accuracy()
loss_fn = torch.nn.CrossEntropyLoss()
log_dir = Path("models/attention/")

from src.models import rnn_models

# de best scorende uit de hypertuning
config = {
    "input_size": len(v),
    "hidden_size": 120,
    "num_layers": 2,
    "dropout": 0.1658799308143347,
    "output_size": 2,
}


model = rnn_models.AttentionNLP(config)
model = train_model.trainloop(
    epochs=100,
    model=model,
    metrics=[accuracy],
    optimizer=torch.optim.Adam,
    learning_rate=0.01,
    loss_fn=loss_fn,
    train_dataloader=trainloader,
    test_dataloader=testloader,
    log_dir=log_dir,
    eval_steps=15,
    train_steps=len(trainloader),
    patience=6,
    factor=0.5,
)
