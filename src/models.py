"""Architectures extracted unchanged from the executed training notebook."""
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

class TransactionMLP(nn.Module):
    def __init__(self, input_dim=93, hidden_dim=64, dropout=0.2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

class TransactionGraphSAGE(nn.Module):
    def __init__(self, input_dim=93, hidden_dim=64, dropout=0.2):
        super().__init__()

        self.conv1 = SAGEConv(
            input_dim, hidden_dim, aggr="mean"
        )

        self.conv2 = SAGEConv(
            hidden_dim, hidden_dim, aggr="mean"
        )

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        return self.output(x).squeeze(-1)
