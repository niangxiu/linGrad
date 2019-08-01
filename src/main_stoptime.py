import network
import mnist_loader
import os.path
import data_generator
import os
import numpy as np
import pickle
from pdb import set_trace
import time


# DIST all run
net_sizes = [50, 50, 50, 50]
data_sizes = [50000, 1, 10000]
training_data, validation_data, test_data = data_generator.load_data(net_sizes,data_sizes)

# # MNIST all run
# # net_sizes = [784, 30, 10]
# # training_data, test_data = mnist_loader.load_data_wrapper(validation = False)

# etas = [None, None, None, None,     None, None, None,   0.01, 0.03, 0.1, 0.3,     1, 3, 10, 30,                 100, 300, 1000]
# epsstars = [0.1, 0.15, 0.2, 0.3,    0.45, 0.67, 1.0,    None, None, None, None,   None, None,  None, None,      None, None,  None]
# mbs = [10, 10, 10, 10,              10, 10, 10,         10, 10, 10,10,            10, 10, 10, 10,               10, 10, 10]
# etas = [None, None, None, None,     None, None, None]
# epsstars = [0.1, 0.15, 0.2, 0.3,    0.45, 0.67, 1.0]
# mbs = [10, 10, 10, 10,              10, 10, 10]
etas =      [None, None, None,  None, None, None, None]
epsstars =  [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
mbs =       [1, 2, 5, 10, 20, 50, 100]
assert len(etas) == len(mbs) == len(epsstars)
nruns = 1
nepoch = 50
all_stoptime = []

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
        j = net.SGD(training_data, epochs=nepoch, mini_batch_size=mb, test_data=test_data, case='DIST', const_eta=eta, target=0.1)
        # _ = net.SGD(training_data, epochs=nepoch, mini_batch_size=mb, test_data=test_data, case='MNIST', const_eta=eta)
        results.append(j)
    results = np.array(results)
    all_stoptime.append(results.mean(axis=0))

all_stoptime = np.array(all_stoptime)
with open("stoptime_batchsize.p", "wb") as f:
    pickle.dump((epsstars, etas, mbs, all_stoptime), f)
