import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

from clean import clean_df
from prepare_sequences import prepare_df, prepare_sequences
from DKT import DKT

csv_to_use = "shorter_training"
dataset = pd.read_csv(f"data/lalilo_datasets/{csv_to_use}.csv")[:500]

cleaned_dataset = clean_df(dataset)
prepared_dataset, label_encoder = prepare_df(cleaned_dataset)

exercise_sequences = prepare_sequences(prepared_dataset)

# hyperparameters
n_epoch = 21
hidden_dim = 40
n_exercise_tuples = label_encoder.classes_.shape[0]

# model, loss_function and optimizer 
model = DKT(n_exercise_tuples, hidden_dim)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1)

for epoch in range(n_epoch):
    for exercise_sequence in exercise_sequences:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # exercise_sequence_columns are "exercise_code_level_lesson", "correctness" 
        # and columns with the one hot encodings of all (exercise_code_level_lesson, correctness) tuples
        sequence_length = len(exercise_sequence)
        exercise_encodings = exercise_sequence["exercise_code_level_lesson"].values
        correctness = torch.tensor(
            exercise_sequence["correctness"].values, dtype=torch.float
        )
        sequence_alone = (
            exercise_sequence.drop(
                columns=["correctness", "exercise_code_level_lesson"]
            )
            .shift() # very important to avoid data leakage
            .fillna(0)
        )
        # forward pass
        predicted_probas = model(
            torch.tensor(sequence_alone.values.astype(int), dtype=torch.float).view(
                sequence_length, 1, 2 * n_exercise_tuples
            )
        ).squeeze()  # shape (sequence_length, n_exercise_tuples)

        # selecting the probas for the exercise that were in fact answered
        predicted_probas_of_answers = predicted_probas[
            np.arange(sequence_length), exercise_encodings
        ]  # shape (sequence_length)

        loss = loss_function(predicted_probas_of_answers, correctness)

        # printing loss
        if epoch % 10 == 1:
            print(loss)
            print()

        # propagating the gradients and updating the weights
        loss.backward()
        optimizer.step()
