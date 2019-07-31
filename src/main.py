import network
import mnist_loader
import os.path
import data_generator
import os
import numpy as np
import pickle
from pdb import set_trace
import time


# # MNIST basic run
# training_data, test_data = mnist_loader.load_data_wrapper(validation = False)
# net = network.Network([784, 30, 10], generator=False, epsstar=0.3)
# net.SGD(training_data, epochs=800, mini_batch_size=10, 
        # test_data=test_data, case='MNIST', const_eta=None)


# # DIST basic run
# net_sizes = [50, 50, 50, 50]
# data_sizes = [50000, 1, 10000]
# training_data, validation_data, test_data = data_generator.load_data(net_sizes,data_sizes)
# net = network.Network(net_sizes, generator=False, epsstar=0.3)
# net.SGD(training_data, epochs=50, mini_batch_size=10, test_data=test_data, case='DIST', const_eta=None)


# DIST all run
# net_sizes = [50, 50, 50, 50]
net_sizes = [50, 50, 50, 50]
data_sizes = [50000, 1, 10000]
training_data, validation_data, test_data = data_generator.load_data(net_sizes,data_sizes)

# MNIST all run
# net_sizes = [784, 30, 10]
# training_data, test_data = mnist_loader.load_data_wrapper(validation = False)

# etas = [None, None, None, None, None,   None, None, None, None, None]
# epsstars = [0.1, 0.2, 0.3, 0.4, 0.5,    0.6, 0.7, 0.8, 0.9, 1.0]
# mbs = [10, 10, 10, 10, 10,              10, 10, 10, 10, 10]
etas = [None, None, None]
epsstars = [1.0, 0.9, 0.8]
mbs = [10, 10, 10]
# etas = [None, None, None, None, None, None, None, None]
# epsstars = [0.1, 0.3, 0.5, 0.8, 0.1, 0.3, 0.5, 0.8]
# mbs = [2, 2, 2, 2, 50, 50, 50, 50]
assert len(etas) == len(epsstars) == len(mbs)
nruns = 1
nepoch = 50
all_hist = []

for eta, epsstar, mb in zip(etas, epsstars, mbs):
    print('eta=', eta, ' eps*=', epsstar, 'minibatch_size=', mb)
    results = []
    for i in range(nruns):
        time.sleep(1)
        try:
            os.remove('checkpoint.p')
        except:
            print("Error while deleting checkpoint")
        try:
            os.remove('record.txt')
        except:
            print("Error while deleting record file")
        net = network.Network(net_sizes, generator=False, epsstar=epsstar)
        j = net.SGD(training_data, epochs=nepoch, mini_batch_size=mb, test_data=test_data, case='DIST', const_eta=eta)
        # _ = net.SGD(training_data, epochs=nepoch, mini_batch_size=mb, test_data=test_data, case='MNIST', const_eta=eta)
        results.append(j)
    results = np.array(results)
    all_hist.append(results.mean(axis=0))

all_hist = np.array(all_hist)
with open("all_history.p", "wb") as f:
    pickle.dump(all_hist, f)
