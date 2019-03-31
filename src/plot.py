import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')


f=open('record.txt',"r")
lines=f.readlines()
eta_hist = []
linerr_hist = []
for x in lines:
    eta_hist.append(float(x.split(' ')[0]))
    linerr_hist.append(float(x.split(' ')[1]))
plt.figure(figsize=[16,12])
plt.plot(eta_hist)
plt.savefig('eta_hist.png')
plt.close()
f.close()

plt.figure(figsize=[16,12])
plt.plot(np.log(linerr_hist))
plt.savefig('linerr_hist.png')
plt.close()
