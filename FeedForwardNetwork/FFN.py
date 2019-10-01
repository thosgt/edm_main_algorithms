import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, n_skills, n_items, n_counters, hidden_dim, drop_prob):
        super(FeedForwardNetwork, self).__init__()
        self.lin_features_to_hidden = nn.Linear(n_counters * (n_items + n_skills), hidden_dim)
        self.lin_hidden_to_output = nn.Linear(hidden_dim, n_items)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, input):
        hidden_state = F.relu(self.lin_features_to_hidden(input))
        output = self.lin_hidden_to_output(self.dropout(hidden_state))
        return output
