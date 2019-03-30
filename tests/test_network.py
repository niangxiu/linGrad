from __future__ import division
import numpy as np
import sys, os
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))
from src.network import sigmoid, sigmoid_prime
import src.network
import src.mnist_loader


def test_run_primal():
    assert 1+1 == 2    

def test_fdld_1():
    """check the finite difference and linearized difference function"""
    net = src.network.Network([2, 1])
    net.biases = [1]
    net.weights = [np.array([1,1])]
    net.eta = 1
    mini_batch = [[np.array([1,1]),1], [np.array([1,1]),1]]
    delta_b = [0.1]
    delta_w = [np.array([0.1, 0.1])]
    fds, lds, eps, eta = net.update_eta(mini_batch, delta_b, delta_w)
    a = sigmoid(3.3) - sigmoid(3)
    assert np.abs(fds - a) <= 1e-9
    b = sigmoid_prime(3) * 0.3
    assert np.abs(lds - b) <= 1e-9
    assert np.abs(eps - np.abs((a-b)/ b)) <= 1e-9
    assert np.abs(eta**2 - src.network.epsstar / eps) <= 1e-9


def test_fdld_2():
    """check the finite difference and linearized difference function"""
    net = src.network.Network([1, 1, 1])
    net.biases = [1, 1]
    net.weights = [np.array([1]), np.array([1])]
    net.eta = 1
    mini_batch = [[np.array([1]),np.array([1])], [np.array([1]),np.array([1])]]
    delta_b = [0.1, 0.1]
    delta_w = [np.array([0.1]), np.array([0.1])]
    fds, lds, eps, eta = net.update_eta(mini_batch, delta_b, delta_w)
    a = np.array([sigmoid(2.2) - sigmoid(2) ,sigmoid(1.1*sigmoid(2.2)+1.1) - sigmoid(sigmoid(2)+1)])
    assert np.all(np.abs(fds - a) <= 1e-9)
    b = np.array([0.2*sigmoid_prime(2), sigmoid_prime(sigmoid(2)+1) * (0.2*sigmoid_prime(2) + 0.1*sigmoid(2)+0.1)])
    assert np.all(np.abs(lds - b) <= 1e-9)
    c = np.abs((a[0]-b[0])/ b[0]) + np.abs((a[1]-b[1])/ b[1])
    c /= 2
    print(c)
    assert np.abs(eps - c) <= 1e-9
    assert np.abs(eta**2 - src.network.epsstar / eps) <= 1e-9

