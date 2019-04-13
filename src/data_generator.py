"""
data_generator
~~~~~~~~~~~~

generate data from one fixed network with randomly generated parameters.
"""

#### Libraries
# Standard library
import pickle
import gzip
import os

# Third-party libraries
import numpy as np
import network


def load_data(net_sizes, data_sizes):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data``, ``validation_data`` and ``test_data``
    are lists containing 2-tuples ``(x, y)``.  
    data_size: list containing the number of training, validation,
    and test data.

    """
    if os.path.exists('data.p'):
        print('reload data for artificial network')
        training_data, validation_data, test_data = pickle.load(open( "data.p", "rb" ))
    else:
        assert len(data_sizes) == 3
        print('no data.p found, generating data for artificial network')
        net = network.Network(net_sizes, generator=True)
        training_data = generate_data(net, data_sizes[0])
        validation_data = generate_data(net, data_sizes[1])
        test_data =  generate_data(net, data_sizes[2])
        with open("data.p", "wb") as data_file:
            pickle.dump((training_data, validation_data, test_data), data_file)
    return (training_data, validation_data, test_data)


def generate_data(net, data_size):
    inputs = [np.random.randn(net.sizes[0], 1) for i in range(data_size)]
    results = [net.feedforward(x) for x in inputs]
    data = zip(inputs, results)
    return data
