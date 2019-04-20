import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')


# # plot history of eta and linearity error of one sample
# f=open('record.txt',"r")
# lines=f.readlines()
# eta_hist = []
# linerr_hist = []
# for x in lines:
    # eta_hist.append(float(x.split(' ')[0]))
    # linerr_hist.append(float(x.split(' ')[1]))
# plt.figure(figsize=[16,12])
# plt.plot(eta_hist)
# plt.savefig('eta_hist.png')
# plt.close()
# f.close()

# plt.figure(figsize=[16,12])
# plt.plot(np.log(linerr_hist))
# plt.savefig('linerr_hist.png')
# plt.close()


# plot history of error for different eta and eps*
with open("all_history.p", "rb") as f:
    all_hist = pickle.load(f)
plt.figure(figsize=(10,8))
plt.plot(all_hist.T)
plt.grid
plt.legend(('\epsilon^*=0.1', '\epsilon^*=0.3', '\eta=0.1', '\eta=1', '\eta=10'),
           loc='upper right')
plt.savefig('all_case_error_hist.png')
plt.close()

