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

