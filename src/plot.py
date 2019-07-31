import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from pdb import set_trace

plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')


# # plot history of eta and linearity error of one sample
# f = open('record_psi_initial_001.txt',"r")
# lines = f.readlines()
# eta_hist = []
# linerr_hist = []
# for x in lines:
    # eta_hist.append(float(x.split(' ')[0]))
    # linerr_hist.append(float(x.split(' ')[1]))
# n = len(eta_hist)
# f.close()

# f = open('record_psi_initial_1.txt',"r")
# lines = f.readlines()
# eta_hist1 = []
# linerr_hist1 = []
# for x in lines:
    # eta_hist1.append(float(x.split(' ')[0]))
    # linerr_hist1.append(float(x.split(' ')[1]))
# n1 = len(eta_hist1)
# assert n1 == n
# f.close()

# plt.figure(figsize=[8,7])
# plt.subplot(211)
# plt.plot(np.arange(n)/50.0, eta_hist)
# plt.plot(np.arange(n1)/50.0, eta_hist1)
# plt.ylabel('$\psi$')
# plt.gca().axes.get_xaxis().set_visible(False)
# plt.subplot(212)
# plt.semilogy(np.arange(n)/50.0, linerr_hist)
# plt.semilogy(np.arange(n1)/50.0, linerr_hist1)
# plt.ylabel('$\epsilon$')
# plt.xlabel('epochs')
# plt.tight_layout()
# plt.legend(('$\psi_0=0.01$','$\psi_0=1$'), loc='lower right')
# plt.savefig('step_nonlinmeas_hist.png')
# plt.close()


# plot history of error for different eta and eps* for DIST/MNIST
# styles = ['y-', 'b-', 'r-', 'g-', 'k-','b:','r:', 'g:', 'k:']
# styles = ['b-', 'r-', 'g-', 'k-']
# styles = ['b-', 'r-', 'g-', 'k-','b:','r:','g:','k:']
with open("all_history.p", "rb") as f:
    all_hist = pickle.load(f)
    # assert len(all_hist) == len(styles)

plt.figure(figsize=(8,6))
# for y, s in zip(all_hist, styles):
    # plt.plot(y, s)
for y in all_hist:
    plt.plot(y)
plt.grid
plt.xlabel('epochs')
plt.ylabel('objective')
# plt.legend(('$\epsilon^*=0.05$', '$\epsilon^*=0.1$', '$\epsilon^*=0.3$', '$\epsilon^*=0.5$',
            # '$\epsilon^*=0.8$', '$\psi=0.1$', '$\psi=1$', '$\psi=10$', '$\psi=100$'),
            # loc='upper right')
plt.legend((
    '$\epsilon^*=0.1$', '$\epsilon^*=0.2$', '$\epsilon^*=0.3$', '$\epsilon^*=0.4$',
    '$\epsilon^*=0.5$', '$\epsilon^*=0.6$', '$\epsilon^*=0.7$', '$\epsilon^*=0.8$',
    '$\epsilon^*=0.9$', '$\epsilon^*=1.0$' 
    ), loc='upper right')
# plt.legend(('$N_s=2$, $\epsilon^*=0.1$', 
            # '$N_s=2$, $\epsilon^*=0.3$', 
            # '$N_s=2$, $\epsilon^*=0.5$', 
            # '$N_s=2$, $\epsilon^*=0.8$', 
            # '$N_s=50$, $\epsilon^*=0.1$', 
            # '$N_s=50$, $\epsilon^*=0.3$', 
            # '$N_s=50$, $\epsilon^*=0.5$', 
            # '$N_s=50$, $\epsilon^*=0.8$'),
            # loc='upper right')
plt.tight_layout()
plt.savefig('DIST_obj_hist.png')
# plt.savefig('MNIST_obj_hist_different_mbsize.png')
plt.close()
