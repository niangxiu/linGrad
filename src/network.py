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


class Network(object):


    def __init__(self, sizes, generator=False, epsstar=1.0, weights_factor=1.0):
        """generator = True if want to generate data from prefixed network"""
        global record_file
        if not generator:
            record_file = open("record.txt", 'w')
        self.N1 = 10 # number of first minibatches to discard
        self.N2 = 30 # minibatchs to remember
        self.N3 = 10 # more batches after switch to select eta by min
        self.epsstar = epsstar
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.eta = 1.0 # initial learning rate 
        self.etas = [] # history of learning rate
        self.obj_factor = 1.0
        self.nepoch = 0
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * weights_factor
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.hist = []


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    

    def test_result(self, test_data, case):
        n_test = len(test_data)
        if case == 'MNIST':
            wrong_perc = self.evaluate_MNIST(test_data)
            print "Epoch {0}: {1} wrong / {2}".format(self.nepoch, wrong_perc, n_test)
            self.hist.append(wrong_perc)
        if case == 'DIST':
            dist = self.evaluate_DIST(test_data)
            print "Epoch {0}: distance {1}".format(self.nepoch, dist)
            self.hist.append(dist)


    def SGD(self, training_data, epochs, mini_batch_size, test_data, case='MNIST'):
        """Train the neural network using mini-batch stochastic gradient descent.  
        case='MNIST' or 'DIST', where DIST is data generated by an prefixed network."""
        n = len(training_data)
        self.test_result(test_data, case)
        for j in xrange(epochs):
            self.nepoch += 1
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for i, mini_batch in enumerate(mini_batches):
                self.update_parameter(mini_batch)
            self.test_result(test_data, case)
                                            

    def parameter_perturb(self, mini_batch):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # delta means coming from one sample
            nabla_b = [nb+dnb/len(mini_batch) for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw/len(mini_batch) for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # update by fastest descent direction
        delta_b = [-(self.eta)*nb for nb in nabla_b]
        delta_w = [-(self.eta)*nw for nw in nabla_w]
        return delta_b, delta_w


    def update_parameter(self, mini_batch):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.  """
        delta_b, delta_w = self.parameter_perturb(mini_batch)
        self.biases  = [b+db for b, db in zip(self.biases,  delta_b)]
        self.weights = [w+dw for w, dw in zip(self.weights, delta_w)]


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function.  nabla_b and
        nabla_w are layer-by-layer lists of numpy arrays, similar to self.biases and self.weights."""
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
        network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return 1- (sum(int(x == y) for (x, y) in test_results)) / len(test_data)


    def evaluate_DIST(self, test_data):
        """Return the mean of the distances between the output and test data"""
        test_results = [(self.feedforward(x), y)
                        for (x, y) in test_data]
        return np.mean([np.linalg.norm(x-y)/np.sqrt(len(y)) for (x, y) in test_results])


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives for the output activations."""
        return (output_activations-y) * self.obj_factor

        
    def finite_difference(self, base_acts, delta_b, delta_w):
        """return the perturbation in the entire network. 
        base_acts is the activations of the network with base parameters"""
        # feedforward with perturbed parameters
        weights = [w+dw for w, dw in zip(self.weights, delta_w)]
        biases  = [b+db for b, db in zip(self.biases,  delta_b)]
        activation = base_acts[0]
        activations_p = [activation]
        for b, w in zip(biases, weights):
            z = np.dot(w, activation)+b
            activation = sigmoid(z)
            activations_p.append(activation)
        return [ap-a for ap, a in zip(activations_p[1:], base_acts[1:])]


    def finite_difference_obj(self, base_acts, delta_b, delta_w, y):
        """return the perturbation in only the objective. 
        base_acts is the activations of the network with base parameters """
        # feedforward with perturbed parameters
        weights = [w+dw for w, dw in zip(self.weights, delta_w)]
        biases  = [b+db for b, db in zip(self.biases,  delta_b)]
        activation = base_acts[0]
        activations_p = [activation]
        for b, w in zip(biases, weights):
            z = np.dot(w, activation)+b
            activation = sigmoid(z)
            activations_p.append(activation)
        return (0.5*np.sum((activations_p[-1]-y)**2) - 0.5*np.sum((base_acts[-1]-y)**2)) * self.obj_factor


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
        

    def update_eta(self, mini_batch, delta_b, delta_w, eps_only_obj=False):
        """compute nonlinear measurement eps and update stepsize eta"""
        eps = 0
        if not eps_only_obj:
            for x, y in mini_batch:
                # feedforward base net
                activation = x
                base_acts = [x] # list to store all the activations, layer by layer
                for b, w in zip(self.biases, self.weights):
                    z = np.dot(w, activation)+b
                    activation = sigmoid(z)
                    base_acts.append(activation)
                # compute linear error
                fds = self.finite_difference(base_acts, delta_b, delta_w)
                lds = self.linearized_perturb(base_acts, delta_b, delta_w)
                epsn = 0.0
                for fd, ld, i in zip(fds, lds, range(len(fds))):
                    # epsn += norm(fd-ld) / (norm(ld) + 1e-12) / len(fds)
                    epsn += norm(fd-ld) / (min(norm(fd), norm(ld)) + 1e-12) /  len(fds)
                    # epsn += norm(fd-ld) / (min(norm(fd), norm(ld)) + 1e-12)
                eps += epsn / len(mini_batch)
        else:
            fdJ = 0
            for x, y in mini_batch:
                activation = x
                base_acts = [x]
                for b, w in zip(self.biases, self.weights):
                    z = np.dot(w, activation)+b
                    activation = sigmoid(z)
                    base_acts.append(activation)
                fdJ +=  self.finite_difference_obj(base_acts, delta_b, delta_w, y) / len(mini_batch)
            nabla_b = [-db/(self.eta) for db in delta_b]
            nabla_w = [-dw/(self.eta) for dw in delta_w]
            ldJ = [np.sum(db*nb)+np.sum(dw*nw) for db, dw, nb, nw in zip(delta_b, delta_w, nabla_b, nabla_w)]
            ldJ = np.sum(ldJ)
            eps = np.abs(fdJ-ldJ) / (min(np.abs(ldJ), np.abs(fdJ)) + 1e-12)

        # update self.eta
        self.etas.append(self.eta * self.epsstar/eps)
        record_file.write('{:e} {:e} {:e}\n'.format(self.eta, eps, self.etas[-1]))
        record_file.flush()
        if len(self.etas) >= self.N2+self.N1:
            self.eta = min(self.etas[-self.N2:])
        else:
            self.eta = self.etas[-1]
        # return fds, lds, eps, self.eta


    def lingrad(self, training_data, mini_batch_size, eps_only_obj=False):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size]
                    for k in xrange(0, (self.N1+self.N2+self.N3)*mini_batch_size, mini_batch_size)]
        for i, mini_batch in enumerate(mini_batches):
            delta_b, delta_w = self.parameter_perturb(mini_batch)
            self.update_eta(mini_batch, delta_b, delta_w, eps_only_obj)


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

