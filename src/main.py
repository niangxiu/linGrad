import network
import mnist_loader
import os.path
import data_generator
import os
import numpy as np
import pickle
from pdb import set_trace


# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 8, 10, test_data=test_data)


# net_sizes = [50, 50, 50, 50]
# data_sizes = [50000, 1, 10000]
# training_data, validation_data, test_data = data_generator.load_data(net_sizes,data_sizes)
# net = network.Network(net_sizes)
# # net.SGD(training_data, epochs=50, mini_batch_size=10, test_data=test_data, case='DIST', const_eta=None)
# net.SGD(training_data, epochs=50, mini_batch_size=10, test_data=test_data, case='DIST', const_eta=1)


data_sizes = [50000, 1, 10000]
net_sizes = [50, 50, 50, 50]
training_data, validation_data, test_data = data_generator.load_data(net_sizes,data_sizes)

etas = [None, None, 0.1, 1, 10]
epsstars = [0.1, 0.3, None, None, None]
nsamples = 10
nepoch = 200
all_hist = []

for eta, epsstar in zip(etas, epsstars):
    print('eta=', eta, ' eps*=', epsstar)
    results = []
    for i in range(nsamples):
        try:
            os.remove('checkpoint.p')
            os.remove('record.txt')
        except:
            print("Error while deleting file")
        net = network.Network(net_sizes, generator=False, epsstar=epsstar)
        _ = net.SGD(training_data, epochs=nepoch, mini_batch_size=10, test_data=test_data, case='DIST', const_eta=eta)
        results.append(_)
    results = np.array(results)
    all_hist.append(results.mean(axis=0))

all_hist = np.array(all_hist)
with open("all_history.p", "wb") as f:
    pickle.dump(all_hist, f)
    
