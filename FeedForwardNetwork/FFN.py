import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, n_skills, n_items, n_counters, hidden_dim, drop_prob):
        super(FeedForwardNetwork, self).__init__()
        self.lin1 = nn.Linear(n_counters(n_items + n_skills), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, n_items)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, input):
        hidden_state = F.relu(self.lin1(input))
        output = F.sigmoid(self.lin2(self.dropout(hidden_state)))
        return output
