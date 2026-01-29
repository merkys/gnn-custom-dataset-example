import numpy
import pandas
import torch
import torch_geometric
import sys

import custom_dataset_from_graph_csv_files

from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.nn import GCNConv

model_file=sys.argv[1]

dataset=custom_dataset_from_graph_csv_files.CustomDataset(root='training_data')
dataset.shuffle()

data_loader=torch_geometric.loader.DataLoader(dataset, batch_size=1)

model=torch.load(model_file, weights_only=False)
# model=model.to(device)

explainer = Explainer(
    model=model,
    algorithm=PGExplainer(epochs=30),
    explanation_type='phenomenon',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='graph',
        return_type='raw',
    ),
)

for epoch in range(30):
    for batch in data_loader:
        loss = explainer.algorithm.train(
            epoch, model, batch, batch.edge_index, target=batch.y)

explanation = explainer(dataset[0].x, dataset[0].edge_index)
