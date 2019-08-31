import network
import mnist_loader
import os.path
import data_generator
import os
import numpy as np
import pickle
from pdb import set_trace
import time
from copy import deepcopy


def compare_stoptime():
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
        print('eta=', eta, ' eps*=', epsstar, 'mini_batch_size=', mb)
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
            net.lingrad(training_data)
            j = net.SGD(training_data, epochs=nepoch, mini_batch_size=mb, test_data=test_data, case='DIST')
            # j = net.SGD(training_data, epochs=nepoch, mini_batch_size=mb, test_data=test_data, case='MNIST')
            results.append(j)
        results = np.array(results)
        all_stoptime.append(results.mean(axis=0))

    all_stoptime = np.array(all_stoptime)
    with open("stoptime_batchsize.p", "wb") as f:
        pickle.dump((epsstars, etas, mbs, all_stoptime), f)



def compare_hist():
    # DIST all run
    net_sizes = [50, 50, 50, 50]
    data_sizes = [50000, 1, 10000]
    training_data, validation_data, test_data = data_generator.load_data(net_sizes,data_sizes)
    # # MNIST all run
    # net_sizes = [784, 30, 10]
    # training_data, test_data = mnist_loader.load_data_wrapper(validation = False)

    etas = [0.01, 0.1, 1, 10, 100]
    mini_batch_size = 10
    nruns = 1
    nepoch = 100
    epsstar = 1.0
    all_hist = []

    for eta in etas:
        print('eta=', eta, ' mini_batch_size=', mini_batch_size)
        results = []
        for i in range(nruns):
            time.sleep(1)
            try: os.remove('record.txt')
            except: print("Error while deleting record file")
            net = network.Network(net_sizes, generator=False, epsstar=epsstar)
            net.eta = eta
            net.SGD(training_data, epochs=nepoch, mini_batch_size=mini_batch_size, test_data=test_data, case='DIST')
            # net.SGD(training_data, epochs=nepoch, mini_batch_size=mb, test_data=test_data, case='MNIST')
            results.append(net.hist)
        all_hist.append(np.array(results).mean(axis=0))
    with open("all_history.p", "wb") as f:
        pickle.dump(all_hist, f)
    


def simple_DIST():
    net_sizes = [50, 50, 50, 50]
    data_sizes = [50000, 1, 10000]
    mini_batch_size = 1
    epsstar = 0.5
    training_data, validation_data, test_data = data_generator.load_data(net_sizes,data_sizes)
    net = network.Network(net_sizes, generator=False, epsstar=epsstar)
    net.N1 = 100
    net.N2 = 300
    # net.obj_factor = 100
    net.lingrad(training_data, mini_batch_size)
    print(net.eta)
    # net.SGD(training_data, epochs=50, mini_batch_size=mini_batch_size, test_data=test_data, case='DIST')



def compare_DIST():
    eps_only_obj=False
    weights_factor = 1.0
    nruns = 1
    nepoch = 10

    data_sizes = [50000, 1, 10000]
    net_sizes_data = [50, 50, 50, 50]
    nlayers = 20
    net_sizes = [50 for i in range (nlayers)] 
    # net_sizes_data = [50,40,30,20]
    # nlayers = 1
    # net_sizes = [50 for i in range (nlayers)] + [40 for i in range(nlayers)] + [30 for i in range(nlayers)] + [20 for i in range(nlayers)]
    
    training_data, validation_data, test_data = data_generator.load_data(net_sizes_data,data_sizes)

    etas = [None, None, None, None, None, None,   0.01, 0.03, 0.1, 0.3,     1, 3, 10, 30,                 100, 300, 1000]
    epsstars = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0,    None, None, None, None,   None, None,  None, None,      None, None, None]
    # etas = [None, None, None, None, None, None]
    # epsstars = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    mbs =           [1, 10, 100,    10, 10]
    obj_factors =   [1, 1, 1,       0.01, 100]
    assert len(etas) == len(epsstars)
    assert all([eta==None or epsstar==None for (eta, epsstar) in zip(etas, epsstars)])
    assert len(mbs) == len(obj_factors)

    all_obj = []
    all_eta_lr = [] # to record etas computed via linear range (linGrad)

    net_base = network.Network(net_sizes, generator=False, weights_factor=weights_factor)
    for obj_factor, mini_batch_size in zip(obj_factors, mbs):
        all_obj.append([])
        all_eta_lr.append([])
        for eta, epsstar in zip(etas, epsstars):
            print('eta=', eta, ' eps*=', epsstar, ' mini_batch_size=', mini_batch_size, ' obj_factor=', obj_factor)
            results = []
            eta_lr = [] 
            for i in range(nruns):
                time.sleep(1)
                try: os.remove('record.txt')
                except: print("Error while deleting record file")
                if epsstar is not None:
                    temp = []
                    for i in range(10):
                        net = deepcopy(net_base)
                        net.epsstar = epsstar
                        net.obj_factor = obj_factor
                        net.lingrad(training_data, mini_batch_size, eps_only_obj=eps_only_obj)
                        temp.append(net.eta)
                        print(net.eta)
                    net.eta = np.mean(temp)
                    print(net.eta)
                    eta_lr.append(net.eta)
                if eta is not None:
                    net = deepcopy(net_base)
                    net.epsstar = epsstar
                    net.obj_factor = obj_factor
                    net.eta = eta
                net.SGD(training_data, epochs=nepoch, mini_batch_size=mini_batch_size, test_data=test_data, case='DIST')
                results.append(net.hist[-1])
            all_obj[-1].append(np.array(results).mean(axis=0))
            if epsstar is not None: all_eta_lr[-1].append(np.array(eta_lr).mean(axis=0))
    
    with open("all_objective.p", 'wb') as f:
        pickle.dump((etas, epsstars, mbs, obj_factors, all_obj, all_eta_lr), f)


def eps_psi0():
    # plot the stepsize given by linGrad starting from different psi0
    psi0s = [10**i for i in range(-10, 11)]
    epsstar = 1.0
    net_sizes = [50, 50, 50, 50, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40, 40, 40, 30, 30, 30, 30, 30, 30, 30, 30, 20, 20, 20, 20, 20, 20, 20, 20,]
    # net_sizes = [50, 50, 50, 50]
    data_sizes = [50000, 1, 10000]
    mini_batch_size = 10
    training_data, validation_data, test_data = data_generator.load_data(net_sizes,data_sizes)
    etas = []
    for psi0 in psi0s:
        net = network.Network(net_sizes, generator=False, epsstar=epsstar)
        net.eta=psi0
        net.lingrad(training_data, mini_batch_size)
        etas.append(net.eta)
        print(net.eta)
    with open("eta_psi0.p", "wb") as f:
        pickle.dump((psi0s, etas), f)


def eps_obj():
    # plot the stepsize given by linGrad using only objective defined nonlinear measurement
    psi0s = [10**i for i in range(-10, 11)]
    psi0s = [1e-10,]
    psi0 = 1.0
    epsstar = 1.0
    net_sizes_data = [50, 35, 20]
    nlayers = 10
    # net_sizes = [50 for i in range (nlayers)] + [35 for i in range(nlayers)] + [20 for i in range(nlayers)]
    # net_sizes = [50, 50, 50, 50]
    net_sizes = [50 for i in range(20)]
    data_sizes = [50000, 1, 10000]
    mini_batch_size = 1
    mbs = [1,10,100]
    training_data, validation_data, test_data = data_generator.load_data(net_sizes_data,data_sizes)
    etas=[]
    etas_obj = []
    for i in range(50):
    # for psi0 in psi0s:
    # for mini_batch_size in mbs:
        net = network.Network(net_sizes, generator=False, epsstar=epsstar)
        net.eta=psi0
        net.lingrad(training_data, mini_batch_size, eps_only_obj=False)
        etas.append(net.eta)
        print(net.eta)
        net = network.Network(net_sizes, generator=False, epsstar=epsstar)
        net.eta=psi0
        net.lingrad(training_data, mini_batch_size, eps_only_obj=True)
        etas_obj.append(net.eta)
        print('only objective', net.eta)
    print('avg:', np.mean(etas), 'std:', np.std(etas))
    print('avg of only obj:', np.mean(etas_obj), 'std of only obj:', np.std(etas_obj))
    # with open("eta_psi0.p", "wb") as f:
        # pickle.dump((psi0s, etas), f)


if __name__ == '__main__':

    # compare_hist()
    # compare_stoptime()
    # simple_DIST()
    compare_DIST()
    # eps_psi0()
    # eps_obj()
