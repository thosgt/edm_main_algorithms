This is an Pytorch implementation of [Deep Knowledge Tracing](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf) for a dataset like Lalilo's.

### Some context

Knowledge Tracing (KT) is measuring the evolving knowledge of a student over time. Usually this knowledge is captured in a vector of numbers. Several algorithms have been tried over the years like Item Response Theory (IRT), Bayesian Knowledge Tracing (BKT) etc.

Recently, new KT models based on Neural Networks were tested. Deep Knowledge Tracing (DKT) was the first among them.

## How Deep Knowledge Tracing works

### High level description 

With any Machine Learning algorithm, the first question is : what do I want to predict ?

Here I want to predict the probability of success of a student to an exercise.

Long-Short-Term Memory neural networks (LSTMs) seem a good way to model this : they possess a hidden state supposedly able to capture the evolving student knowledge with each answer.
The hidden state is comprised of ```n_hidden``` dimensions representing the estimated student knowledge state.

From the student knowledge state we should then be able to predict the probability of success of a student to any exercise after any of their answer.

#### More advanced paragraph

A central hypothesis of this algorithm is : the hidden state transitions are the same for each students. Training the model is computing the matrix governing the hidden state transitions.

### Lower level description - if you are familiar with Machine Learning in general

There are two kind of matrices that we need to distinguish here :
- the hidden state that is specific to a given student. There is one hidden state per student. In the beginning of their exercise sequence, the hidden state of a student is a matrix with zeros. It is updated after each exercise they answer
- the weights of the network that govern the transition between hidden states, and the mapping between hidden states and predicted probabilities. These weights are the same for all students. There are updated during training so that they fit student transitions the best way possible : training the model is updating these weights

#### What are we going to feed our network to train it ?
We are going to feed our network the exercise sequence  of each student one after the other (indeed it is a sequence as the exercises are done one after the other and not simultaneously). Therefore, for each student :
- we select the exercises and answers of this student
- the hidden state of the student is set to a zero-like vector
- then for each exercise of their exercise sequence :
  - using the hidden state of the student and the mapping between hidden state and expected probabilities, we predict the probability of answering correctly to the exercise they get and compare it to the actual correctness
  - we update the network weights so that the predicted probability is closer to the actual correctness
  - the hidden state of the student is updated

This is how we train the model to find its weights.

#### Lowest level - if you are already familiar with recurring neural networks in particular

What kind of input my algorithm takes and what is the kind of output it outputs ?

For each student, the input size will vary as the number of exercises each of them answered vary. We set the number of exercises a student answered as ```sequence_length```

One central question here is : how to represent an answer to one of the exercises ?

The solution choosen in the article is to [one-hot-encode](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) the tuple ```(exercise, correctness)```.
For each answer the student does one out of 2 x ```n_available_exercises``` actions : they get one of ```n_available_exercises``` exercises and can answer correctly or not. Thus, for each answer, the input is a one-hot-encoded vector of size ```2 x n_available_exercises```

As output, we would like to have the probabilities of success of all available exercises for the *next* exercise the student does. Therefore, for each answer of the student, the LSTM outputs ```n_available_exercises``` probabilities

When the student hasn't answered any exercise, we still want to have some probabilities from the LSTM. These probabilities would be the starting probabilities for all students when there is no hidden state yet. However the LSTM doesn't output something when it doesn't have an input. The solution we found is to have as the first input of the sequence a row full of zeros i.e having the answer shifted.

To sum up, the input of the LSTM for each student is of shape ```(sequence_length,  2 x n_available_exercises)```and the output is ```(sequence_length,  n_available_exercises)```

Computing the loss is straightforward once we have the predictions for every available exercise and for every actual answer. For each anwer, we need to select the prediction that is relevant (we need only one of the ```n_available_exercises``` predictions, the prediction of the exercise that was actually answered) and compute the ```log_loss``` between it and its actual correctness.

After propagating the gradient and updating the weights we can go to another student sequence and update the weights once more.


### FAQ
##### Why not feeding the entire sequences of answers of all students to the neural network ?
The hidden state is specific to each student so it has to be zeroed between students. That is why the network is fed sequence by sequence.