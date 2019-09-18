import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DKT(nn.Module):
    def __init__(self, n_exercise_tuples, hidden_dim):
        super(DKT, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(n_exercise_tuples * 2, hidden_dim)
        # input is 2 * n_tuples because answers are encoded as (exercise, correctness)

        self.hidden2prediction = nn.Linear(hidden_dim, n_exercise_tuples)
        # the linear layer maps from hidden state space to success_rate estimation on exercises

        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence_of_answer): 
        # sequence of answer is of shape (sequence_length, n_exercise_tuples * 2)
        lstm_out, _ = self.lstm(sequence_of_answer) 
        # lstm_out is of shape (sequence_length, hidden_dim)
        predicted_values = self.hidden2prediction(lstm_out) 
        # predicted_values is of shape (sequence_length, n_exercise_tuples)
        predicted_probas = self.sigmoid(predicted_values)
        return predicted_probas

