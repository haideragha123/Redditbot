import numpy as np

class RNN:

	def step(self, x):

		#update the hidden state
		self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))

		#compute the output vector
		y = np.dot(self.W_hy, self.h)
		return y


	y1 = rnn1.step(x)
	y = rnn2.step(y1)


data = ''
chars = list(set(data))

data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
"""
inputs,targets are both list of integers.
hprev is Hx1 array of initial hidden state
returns the loss, gradients on model parameters, and last hidden state
"""

	xs, hs, ys, ps = {}, {}, {}, {}
	hs[-1] = np.copy(hprev)
	loss = 0
	# forward pass
	for t in xrange(len(inputs)):
		xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
		xs[t][inputs[t]] = 1
		hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
		ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
		loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

	# backward pass: compute gradients going backwards
	dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
	dbh, dby = np.zeros_like(bh), np.zeros_like(by)
	dhnext = np.zeros_like(hs[0])
	for t in reversed(xrange(len(inputs))):
		dy = np.copy(ps[t])
		dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
		dWhy += np.dot(dy, hs[t].T)
		dby += dy
		dh = np.dot(Why.T, dy) + dhnext # backprop into h
		dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
		dbh += dhraw
		dWxh += np.dot(dhraw, xs[t].T)
		dWhh += np.dot(dhraw, hs[t-1].T)
		dhnext = np.dot(Whh.T, dhraw)
	for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
		np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
	return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]