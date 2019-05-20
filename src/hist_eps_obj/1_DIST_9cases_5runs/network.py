"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent with learning
rate given by effective range.
"""

#### Libraries
# Standard library
from __future__ import division
import random
import sys
from pdb import set_trace
import os.path

# Third-party libraries
import numpy as np
from numpy.linalg import norm
from itertools import izip
from scipy.stats.mstats import gmean
import pickle
from pdb import set_trace


# parameters
Nlin = 10 # try to adjust eta this many minibatches
# Nlin = 1 # try to adjust eta this many minibatches


class Network(object):


    def __init__(self, sizes, generator=False, epsstar=0.3):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron. 
        Automatically resume if the pickle file exists.
        generator = True if want to generate data from prefixed network"""
        global record_file
        if os.path.exists('checkpoint.p') and not generator:
            print('resume from checkpoint')
            self.__dict__.clear()
            self.__dict__.update(pickle.load(open( "checkpoint.p", "rb" )))
            record_file = open("record.txt", 'a')
            assert self.sizes == sizes
        else:
            if not generator:
                print('no checkpoint found, fresh start')
                record_file = open("record.txt", 'w')
                print('open record.txt to write')
            self.epsstar = epsstar
            self.Nhist = None # number of previous effective ranges to remember
            # self.Nhist = 50 # default, update in SGD
            self.num_layers = len(sizes)
            self.sizes = sizes
            self.eta = 0.001 # learning rate 
            self.etas = [] # history of learning rate
            self.nepoch = 0
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


    def SGD(self, training_data, epochs, mini_batch_size, test_data=None, case='MNIST', const_eta=None):
        """Train the neural network using mini-batch stochastic gradient descent.  
        N_s = mini_batch_size
        If provided const_eta then we use constant learning rate, if None then do linGrad.
        case='MNIST' or 'DIST', where DIST is data generated by an prefixed network."""
        if test_data: n_test = len(test_data)
        if const_eta is not None: self.eta = const_eta
        n = len(training_data)
        self.Nhist = max(50, int(round(n / Nlin / mini_batch_size)) ) # number of previous effective ranges to remember
        
        results = []
        if case == 'MNIST':
            wrong_perc = self.evaluate_MNIST(test_data)
            print "Epoch 0: {} wrong / {}".format(wrong_perc, n_test)
            results.append(wrong_perc)
        if case == 'DIST':
            dist = self.evaluate_DIST(test_data)
            print "Epoch 0: distance {}".format(dist)
            results.append(dist)
        
        for j in xrange(epochs):
            self.nepoch += 1
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for i, mini_batch in enumerate(mini_batches):
                if i % Nlin == 0 and const_eta is None:
                    self.update_mini_batch(mini_batch, effectrange = True)
                else:
                    self.update_mini_batch(mini_batch)
            if test_data:
                if case == 'MNIST':
                    wrong_perc = self.evaluate_MNIST(test_data)
                    print "Epoch {0}: {1} wrong / {2}".format(
                        self.nepoch, wrong_perc, n_test)
                    results.append(wrong_perc)
                if case == 'DIST':
                    dist = self.evaluate_DIST(test_data)
                    print "Epoch {0}: distance {1}".format(
                        self.nepoch, dist)
                    results.append(dist)
            else:
                print "Epoch {0} complete".format(j)
        with open("checkpoint.p", "wb") as checkpoint_file:
            pickle.dump(self.__dict__, checkpoint_file)
        record_file.close()
        return results


    def update_mini_batch(self, mini_batch, effectrange=False):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``
        if effectrange then update stepsize eta"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # delta means coming from one sample
            nabla_b = [nb+dnb/len(mini_batch) for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw/len(mini_batch) for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # update by fastest descent direction
        delta_b = [-(self.eta)*nb for nb in nabla_b]
        delta_w = [-(self.eta)*nw for nw in nabla_w]
        # update eta by computing effective range
        if effectrange:
            self.update_eta(mini_batch, delta_b, delta_w, nabla_b, nabla_w)
        self.biases  = [b+db for b, db in zip(self.biases,  delta_b)]
        self.weights = [w+dw for w, dw in zip(self.weights, delta_w)]


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def evaluate_MNIST(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return 1- (sum(int(x == y) for (x, y) in test_results)) / len(test_data)


    def evaluate_DIST(self, test_data):
        """Return the summationof the distances between the output
        data and the output activations."""
        test_results = [(self.feedforward(x), y)
                        for (x, y) in test_data]
        return np.mean([np.linalg.norm(x-y)/np.sqrt(len(y)) for (x, y) in test_results])


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

        
    def finite_difference(self, base_acts, delta_b, delta_w, y, entire_trajec=True):
        """return the perturbation in the entire network. 
        base_acts is the activations of the network with base parameters
        entire_trajec: whether use nonlinear measurement of entire trajec or only objec"""
        # feedforward with perturbed parameters
        weights = [w+dw for w, dw in zip(self.weights, delta_w)]
        biases  = [b+db for b, db in zip(self.biases,  delta_b)]
        activation = base_acts[0]
        activations_p = [activation]
        for b, w in zip(biases, weights):
            z = np.dot(w, activation)+b
            activation = sigmoid(z)
            activations_p.append(activation)
        if entire_trajec:
            return [ap-a for ap, a in zip(activations_p[1:], base_acts[1:])]
        else:
            return 0.5*np.sum((activations_p[-1]-y)**2) - 0.5*np.sum((base_acts[-1]-y)**2) 


    def linearized_perturb(self, base_acts, delta_b, delta_w):
        """return the linearized perturbation in the entire network,
        computed by tangent equations"""
        da = np.zeros(base_acts[0].shape) 
        das = [da]
        for a, a_next, w, dw, db in izip(base_acts[:-1], base_acts[1:], self.weights, delta_w, delta_b):
            sp = sigmoid_prime_a(a_next)
            da = (np.dot(w,da) + np.dot(dw,a) + db) * sp 
            das.append(da)
        return das[1:]
        
    
    def update_eta(self, mini_batch, delta_b, delta_w, nabla_b, nabla_w):
        """compute nonlinear measurement eps and update stepsize eta"""
        eps = 0
        fdJ = 0
        for x, y in mini_batch:
            # feedforward base net
            activation = x
            base_acts = [x] # list to store all the activations, layer by layer
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                activation = sigmoid(z)
                base_acts.append(activation)
            # compute nonlinear measurement from entire network
            # fds = self.finite_difference(base_acts, delta_b, delta_w, y)
            # lds = self.linearized_perturb(base_acts, delta_b, delta_w)
            # epsn = 0.0
            # for fd, ld in zip(fds, lds):
                # epsn += norm(fd-ld) / (norm(ld) + 1e-12) / len(fds)
            # eps += epsn / len(mini_batch)

            # compute nonlinear measurement from only the objective
            fdJ +=  self.finite_difference(base_acts, delta_b, delta_w, y, entire_trajec = False)
        fdJ /= len(mini_batch)
        ldJ = [np.sum(db*nb)+np.sum(dw*nw) for db, dw, nb, nw in zip(delta_b, delta_w, nabla_b, nabla_w)]
        ldJ = np.sum(ldJ)
        eps = np.abs(fdJ-ldJ) / (np.abs(ldJ) + 1e-12)

        # update self.eta
        self.etas.append(self.eta * self.epsstar/eps)
        if len(self.etas) >= self.Nhist:
            self.eta = min(self.etas[-self.Nhist:])
        else:
            self.eta = min(self.etas)
        record_file.write('{:f} {:f}\n'.format(self.eta, eps))
        record_file.flush()
        # return fds, lds, eps, self.eta


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def sigmoid_prime_a(a):
    """Derivative of the sigmoid function, using a instead of z"""
    return a*(1-a)
