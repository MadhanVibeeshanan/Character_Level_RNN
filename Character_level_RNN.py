# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:24:11 2018

@author: library
"""

import operator
import numpy as np
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

"""
filenames = ['C:/Users/vibeeshm/Desktop/Assignment 4/1268-0.txt', 'C:/Users/vibeeshm/Desktop/Assignment 4/pg18857.txt', 'C:/Users/vibeeshm/Desktop/Assignment 4/pg103.txt']
with open('C:/Users/vibeeshm/Desktop/Assignment 4/text_file.txt', 'w', encoding="utf8") as outfile:
    for fname in filenames:
        with open(fname, encoding="utf8") as infile:
            for line in infile:
                outfile.write(line)
"""

#read the text file
#concat_text_file = 'C:/Users/library/Documents/Madhan Deep Learning Assignment/Assignment 4/text_file.txt'

concat_text_file = 'C:/Users/library/Documents/Madhan Deep Learning Assignment/Assignment 4/1268-0.txt'


import nltk.data
from nltk import tokenize

with open(concat_text_file, 'r', encoding = 'utf8') as f:
    data = f.read()

tokenized_data = tokenize.word_tokenize(data)
#tokenized_sentences = [nltk.word_tokenize(sent) for sent in data]

vocabulary_size = 88
unknown_token = "@"
sentence_start_token = "#"
sentence_end_token = "&"


tokenized_data = ["%s%s%s" % (sentence_start_token, x, sentence_end_token) for x in tokenized_data]
tokenized_data.append("# &")

ix_to_char = {i:ch for i, ch in enumerate(list(set(data)))}
char_to_ix = {ch:i for i, ch in enumerate(list(set(data)))}

#XTrain = np.asarray([[char_to_ix[w] for w in sent[:-1]] for sent in chars])
#yTrain = np.asarray([[char_to_ix[w] for w in sent[1:]] for sent in chars])


XTrain = np.asarray([[char_to_ix[w] for w in sent[:-1]] for sent in tokenized_data])
yTrain = np.asarray([[char_to_ix[w] for w in sent[1:]] for sent in tokenized_data])

class RNNVanilla:
     
    def __init__(self, word_dim, hidden_dim=200, bptt_truncate=4):
        
        # Assign instance variables
        self.word_dim = word_dim   #size of the vocabulary
        self.hidden_dim = hidden_dim  # size of hidden layer
        self.bptt_truncate = bptt_truncate
        
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def forward_propagation(self, x):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
        
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
        
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros((T, self.word_dim))
        
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]))
    return [o, s]  #We not only return the calculated outputs, but also the hidden states. 
                   #We will use them later to calculate the gradients

#Now make it a member of the class RNNVanilla
RNNVanilla.forward_propagation = forward_propagation

def predict(self, x):
    # Perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x)
    return np.argmax(o, axis=1)

#Now make it a member of the class RNNVanilla
RNNVanilla.predict = predict

np.random.seed(10)
model = RNNVanilla(vocabulary_size)
o, s = model.forward_propagation(XTrain[10])
print(o.shape)
print(o)

#The following gives the indices of the highest probability predictions for each word:
predictions = model.predict(XTrain[10])
print(predictions.shape)
print(predictions)
print("index_to_word>")
print('%s'%"".join([ix_to_char[x] for x in predictions]))

def calculate_total_loss(self, x, y):
    L = 0
    
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])
        
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        
        # Add to the loss based on how off we were
        L += -1 * sum(np.log(correct_word_predictions))
    return L
 
def calculate_loss(self, x, y):
    # Divide the total loss by the number of training examples
    N = sum((len(y_i) for y_i in y))
    return self.calculate_total_loss(x,y)/N
 
RNNVanilla.calculate_total_loss = calculate_total_loss
RNNVanilla.calculate_loss = calculate_loss

np.log(vocabulary_size)

XTrain[:100].shape

XTrain.shape

# Limit to 1000 examples to save time
print("Expected Loss for random predictions: %f" % np.log(vocabulary_size))
print("Actual loss: %f" % model.calculate_loss(XTrain[:1000], yTrain[:1000]))

def bptt(self, x, y):
    T = len(y)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step-1])              
            dLdU[:,x[bptt_step]] += delta_t
            
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]
 
RNNVanilla.bptt = bptt

def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
    # Calculate the gradients using backpropagation. We want to checker if these are correct.
    bptt_gradients = self.bptt(x, y)
    
    # List of all parameters we want to check.
    model_parameters = ['U', 'V', 'W']
    
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter = operator.attrgetter(pname)(self)
        print ("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            gradplus = self.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            gradminus = self.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            # Reset parameter to original value
            parameter[ix] = original_value
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print ("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                print ("+h Loss: %f" % gradplus)
                print ("-h Loss: %f" % gradminus)
                print ("Estimated_gradient: %f" % estimated_gradient)
                print ("Backpropagation gradient: %f" % backprop_gradient)
                print ("Relative Error: %f" % relative_error)
                return
            it.iternext()
        print ("Gradient check for parameter %s passed." % (pname))
 
RNNVanilla.gradient_check = gradient_check

# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
grad_check_vocab_size = 100
np.random.seed(10)
model = RNNVanilla(grad_check_vocab_size, 10, bptt_truncate=1000)
model.gradient_check([0,1,2,3], [1,2,3,4])

def numpy_sdg_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW
    
RNNVanilla.sgd_step = numpy_sdg_step

# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
loss_for_plot = []
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=10, evaluate_loss_after=2):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        print(epoch,"th epoch")
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            loss_for_plot.append(loss)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print ("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5 
                print ("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
            
np.random.seed(10)
model = RNNVanilla(vocabulary_size)
#print(%timeit model.sgd_step(XTrain[10], yTrain[10], learning_rate=0.005))

np.random.seed(10)
# Train on a small subset of the data to see what happens
model = RNNVanilla(vocabulary_size)
losses = train_with_sgd(model, XTrain, yTrain, nepoch=10, evaluate_loss_after=2)

"""predicitons_list = []
for i in range(len(XTrain)):
    print(i,"th Iteration")
    predictions = model.predict(XTrain[i])
    predicitons_list.append('%s'%"".join([ix_to_char[x] for x in predictions]))
    #print(predictions.shape)
#print(predictions)
#print("index_to_word>")
#print('%s'%"".join([ix_to_char[x] for x in predictions]))

print(losses)
"""

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [char_to_ix[sentence_start_token]]
    
    # Repeat until we get an end token
    while not new_sentence[-1] == char_to_ix[sentence_end_token]:
        next_word_probs, next_word_probs1 = model.forward_propagation(new_sentence)
        sampled_word = char_to_ix[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == char_to_ix[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [ix_to_char[x] for x in new_sentence[1:-1]]
    return sentence_str

num_sentences = 1000
senten_min_length = 1
 
for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print ("".join(sent), end = " ")

epoch_count = range(0, 10, 2)
# Visualize loss history
plt.plot(epoch_count, loss_for_plot, 'r--')
plt.legend(['Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

