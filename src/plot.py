# plot the stopping time for different eta and eps* for DIST

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from pdb import set_trace
from matplotlib.lines import Line2D


plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')


def plot_stoptime():


    with open("stoptime_stepsize.p", "rb") as f:
        all_epsstars, all_etas, mbs, all_stoptime = pickle.load(f)
    n = len(all_epsstars)

    epsstars = [all_epsstars[i] for i in range(n) if all_epsstars[i] is not None]
    stoptimes = [all_stoptime[i] for i in range(n) if all_epsstars[i] is not None]
    fig = plt.figure(figsize=(5,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(epsstars, stoptimes, 'r-', marker='.')
    plt.xscale('log')
    plt.xlabel('$\epsilon^*$')
    plt.ylabel('stopping epoch')

    etas = [all_etas[i] for i in range(n) if all_etas[i] is not None]
    stoptimes = [all_stoptime[i] for i in range(n) if all_etas[i] is not None]
    etas = [0.001,] + etas + [10000,]
    stoptimes = [49,] + stoptimes + [49,]
    ax2 = ax1.twiny()
    ax2.plot(etas, stoptimes, 'k:.')
    plt.xscale('log')
    plt.xlabel('$\psi$')
    plt.ylabel('stopping epoch')

    custom_lines = [Line2D([0], [0], color='r', ls='-', marker='.'),
                    Line2D([0], [0], color='k', ls=':', marker='.')]
    plt.legend(custom_lines, ['linGrad', 'SGD'], loc='center right')
    plt.tight_layout()
    plt.savefig('DIST_stop_time_stepsize.png')
    plt.close()



    # with open("stoptime_batchsize.p", "rb") as f:
        # all_epsstars, all_etas, all_mbs, all_stoptime = pickle.load(f)
    # n = len(all_epsstars)

    # epsstars = [all_epsstars[i] for i in range(n) if all_epsstars[i] is not None]
    # stoptimes = [all_stoptime[i] for i in range(n) if all_epsstars[i] is not None]
    # mbs = [all_mbs[i] for i in range(n) if all_epsstars[i] is not None]
    # plt.figure(figsize=(5,4))
    # plt.plot(mbs, stoptimes, 'r-', marker='.')

    # etas = [all_etas[i] for i in range(n) if all_etas[i] is not None]
    # stoptimes = [all_stoptime[i] for i in range(n) if all_etas[i] is not None]
    # mbs = [all_mbs[i] for i in range(n) if all_etas[i] is not None]
    # plt.plot(mbs, stoptimes, 'k:.')
    # plt.xscale('log')

    # plt.grid
    # plt.xlabel('batch size')
    # plt.ylabel('stopping epoch')
    # plt.ylim(bottom=0)
    # plt.legend(('linGrad', 'SGD'), loc='center right')
    # plt.tight_layout()
    # plt.savefig('DIST_stop_time_batchsize.png')
    # plt.close()


def plot_lr_hist():
    # plot history of eta and linearity error of one sample
    f = open('record_psi_initial_001.txt',"r")
    lines = f.readlines()
    eta_hist = []
    linerr_hist = []
    for x in lines:
        eta_hist.append(float(x.split(' ')[0]))
        linerr_hist.append(float(x.split(' ')[1]))
    n = len(eta_hist)
    f.close()

    f = open('record_psi_initial_1.txt',"r")
    lines = f.readlines()
    eta_hist1 = []
    linerr_hist1 = []
    for x in lines:
        eta_hist1.append(float(x.split(' ')[0]))
        linerr_hist1.append(float(x.split(' ')[1]))
    n1 = len(eta_hist1)
    assert n1 == n
    f.close()

    plt.figure(figsize=[8,7])
    plt.subplot(211)
    plt.plot(np.arange(n)/50.0, eta_hist)
    plt.plot(np.arange(n1)/50.0, eta_hist1)
    plt.ylabel('$\psi$')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.subplot(212)
    plt.semilogy(np.arange(n)/50.0, linerr_hist)
    plt.semilogy(np.arange(n1)/50.0, linerr_hist1)
    plt.ylabel('$\epsilon$')
    plt.xlabel('epochs')
    plt.tight_layout()
    plt.legend(('$\psi_0=0.01$','$\psi_0=1$'), loc='lower right')
    plt.savefig('step_nonlinmeas_hist.png')
    plt.close()


def plot_hist():
    # plot training histories of error for different eta and eps* for DIST/MNIST
    with open("all_history.p", "rb") as f:
        all_hist = pickle.load(f)
    plt.figure(figsize=(8,6))
    for y in all_hist:
        plt.plot(y)
    plt.xlabel('epochs')
    plt.ylabel('objective')
    plt.legend(('$\eta=0.01$', '$\eta=0.1$', '$\eta=1$',
                '$\eta=10$', '$\eta=100$'),
                loc='upper right')
    plt.tight_layout()
    plt.savefig('DIST_obj_hist.png')
    # plt.savefig('MNIST_obj_hist.png')
    plt.close()



def plot_compare_DIST():
    with open("all_objective.p", 'rb') as f:
        etas, epsstars, mbs, obj_factors, all_obj, all_eta_lr = pickle.load(f)
    markers = ['o', 'v', 'x', '+', '1']
    linestyles = ['dashdot', '--', '-', ':', 'dashdot']
    plt.figure(figsize=(8,6))
    lines = []
    for i in range(len(mbs)):
        n = len(etas)
        etas_SGD = [etas[j] for j in range(n) if etas[j] is not None]
        objs_SGD = [all_obj[i][j] for j in range(n) if etas[j] is not None]
        etas_lr = all_eta_lr[i]
        objs_lr = [all_obj[i][j] for j in range(n) if etas[j] is None] 
        lines.append(plt.plot(etas_SGD, objs_SGD, color='k', linestyle=linestyles[i], marker=markers[i], ms=8)[0])
        plt.plot(etas_lr, objs_lr, color='r', linestyle=linestyles[i], marker=markers[i], ms=8)
    plt.xscale('log')
    plt.xlabel('stepsize')
    # plt.ylim(top=0.5)
    plt.ylabel('objective after 50 epochs')
    plt.legend(lines, ['$N_s=1$', 
                '$N_s=10$', 
                '$N_s=100$', 
                'obj_factor=0.01', 
                'obj_factor=100'],
                loc='upper left')
    plt.tight_layout()
    plt.savefig('DIST.png')
    plt.close()
   


def plot_eps_psi0():
    with open("eta_psi0.p", "rb") as f:
        psi0s, etas = pickle.load(f)
    plt.figure(figsize=(8,4))
    plt.plot(psi0s, etas)
    plt.xlabel("$\psi_0$")
    plt.xscale('log')
    plt.ylabel("stepsize given by linGrad")
    plt.yscale('log')
    plt.ylim([1e-1,1e2])
    plt.tight_layout()
    plt.savefig("DIST_eta_psi0.png")
    plt.close()



if __name__ == '__main__':
    # plot_hist()
    # plot_stoptime()
    plot_compare_DIST()
    # plot_eps_psi0()

