This is an Pytorch implementation of [Deep Knowledge Tracing](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf) for a dataset like Lalilo's.

## How Deep Knowledge Tracing works

### High level description

With any Machine Learning algorithm, the first question is : what do I want to predict ?

Here I want to predict the probability of success of a student to an exercise.

Long-Short-Term Memory neural networks (LSTMs) seem a good way to model this : they possess a hidden state supposedly able to capture the evolving student knowledge state with each answer.
The hidden state is comprised of *n_hidden* dimensions representing the estimated student knowledge state.

From the student knowledge state we should then be able to predict the probability of success of a student to any exercise after any of their answer.


#### More advanced paragraph

A central hypothesis of this algorithm is : the hidden state transitions are the same for each students. Training the model is computing the matrix governing the hidden state transitions.


### Lower level description

As we want to trace the evolving knowledge of a student, we will consider student sequences of answers.

The next question is : what kind of input my algorithm takes and what is the kind of output it outputs ?

LSTMs take an input with the ```sequence_length``` as the first dimension size.
The first dimension size of the output will thus also be ```sequence_length```.

The main question here is the input : how to represent an answer to one of the exercises ?

The solution choosen in the article is to [one-hot-encode](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) the tuple ```(exercise, correctness)```.
As correctness can only take two values, the second dimension size of the input is twice the second dimension size of the output.
Rephrasing it, at each timestep the student does one out of 2 x ```n_available_exercises``` actions : they get one of ```n_available_exercises``` exercises and can answer correctly or not.

As output, we would like to have the probabilities of success of all available exercises for the *next* exercise the student does.
The second dimension of the output is ```n_available_exercises```.

When the student hasn't answered any exercise, we still want to have some probabilities from the LSTM. These probabilities would be the starting probabilities for all students when there is no hidden state yet. However the LSTM doesn't output something when it doesn't have an input. The solution we found is to have as the first input of the sequence a row full of zeros i.e having the answer shifted.

The LSTM runs through the complete sequence of a student and returns the estimated probabilities for each of the ```n_available_exercises```exercises at each timestep. Therefore the output of the LSTM is of shape ```(sequence_length, n_available_exercises)```

Computing the loss is straightforward once we have the predictions for every exercise and for every timestep, we need to select at each time step the prediction that is relevant (we need only one of the ```n_available_exercises``` predictions, the prediction of the exercise that was actually answered) and compute the ```log_loss``` between it and its actual correctness.

After propagating the gradient and updating the weights we can go to another student sequence and update the weights once more.