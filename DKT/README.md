This is an Pytorch implementation of [Deep Knowledge Tracing](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf) for a dataset like Lalilo's.

## How Deep Knowledge Tracing works

The LSTM in itself is quite simple (see the DKT class)

What is more complicated is to correctly compute the loss.
Let's say a student answers T exercises (they can answer twice the same kind of exercise).
T is the sequence_length.


There are N available exercises overall.

So at each timestep the student does one out of 2 x N actions : they get one of N exercises and can answer correctly or not.

We thus one-hot-encode these 2 x N possible actions and that is why the input dimension of the LSTM is 2 x N.
The output we want from the LSTM at each timestep is the probabilty to answer correctly one of the N exercises at the **next** timestep.

When the student hasn't answered any exercise, we still want to have some probas from the LSTM. Therefore we shift their answer one timestep and get the "starting" probabilities of students - this is kind of the initial state of students.

The LSTM runs through the complete sequence of a student and returns the estimated probabilities for each of the N exercise at each timestep. Therefore the output of the LSTM is of shape (sequence_length, N)

The loss is computed between the correctness of a student answer and what their estimated probability of success for this exercise.