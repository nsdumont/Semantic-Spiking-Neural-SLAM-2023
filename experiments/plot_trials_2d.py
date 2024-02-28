from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import rc
import numpy as np
import sys
import zipfile
import pandas as pd
import os
from matplotlib import cm
import seaborn as sns

import figure_utils as utils


def read(path):
    data = []
    if os.path.exists(path):
        for fn in os.listdir(path):
            fn = os.path.join(path, fn)
            if fn.endswith('.zip'):
                z = zipfile.ZipFile(fn)
                for name in z.namelist():
                    try:
                        if name.endswith('.txt'):
                            data.append(text(z.open(name)))
                        elif name.endswith('.npz'):
                            data.append(npz(z.open(name)))
                    except:
                        print('Error reading file "%s" in "%s"' % (name, fn))
                z.close()
            else:
                try:
                    if fn.endswith('.txt'):
                        data.append(text(fn))
                    elif fn.endswith('.npz'):
                        data.append(npz(fn))
                except:
                    print('Error reading file "%s"' % fn)
    return data

def text(fn):
    if not hasattr(fn, 'read'):
        with open(fn) as f:
            text = f.read()
    else:
        text = fn.read()
    d = {}
    try:
        import numpy
        d['array'] = numpy.array
        d['nan'] = numpy.nan
    except ImportError:
        numpy = None
    exec(text, d)
    del d['__builtins__']
    if numpy is not None:
        if d['array'] is numpy.array:
            del d['array']
        if d['nan'] is numpy.nan:
            del d['nan']
    return d


def npz(fn):
    import numpy as np
    d = {}
    f = np.load(fn,allow_pickle=True)
    for k in f.files:
        if k!='ssp_space':
            d[k] = f[k]
            if d[k].shape == ():
                d[k] = d[k].item()
    return d


def plot_paths(folder,nrows,ncols,idxs, savename=None):
    data = pd.DataFrame(read(folder))
    ts = data['ts'].values[0]
    n_trials = data.shape[0]
    fig, axs = plt.subplots(figsize=(7, nrows*7/ncols), nrows=nrows,ncols=ncols)
    startidx = 200
    axs=axs.reshape(-1)
    for j in range(len(idxs)):
        i=idxs[j]
        axs[j].plot(data['path'].values[i][:,0], data['path'].values[i][:,1],
                    '-', color=utils.grays[2], label='Ground\n truth', linewidth=1.5)
        axs[j].plot(data['pi_sim_path'].values[i][startidx:,0], 
                    data['pi_sim_path'].values[i][startidx:,1], 
                    '--', color=utils.oranges[0],label='PI', linewidth=1.5)
        axs[j].plot(data['slam_sim_path'].values[i][startidx:,0], 
                    data['slam_sim_path'].values[i][startidx:,1],
                    '--', color=utils.blues[0],label='SLAM', linewidth=1.5)
        axs[j].set_xlim([-1.1,1.1])
        axs[j].set_ylim([-1.1,1.1])
        #axs[j].set_aspect('equal')
        if j != ncols*(nrows-1):
            axs[j].set_xticklabels([])
            axs[j].set_yticklabels([])
        if j == ncols-1:
            axs[j].legend(loc='lower right')#,frameon=True)
        # ax[j].set_ylabel('$y$')
        # ax[j].set_xlabel('$x$')
        axs[j].spines['right'].set_visible(True)
        axs[j].spines['top'].set_visible(True)
    
    fig.subplots_adjust(wspace=0, hspace=0)
    
    if savename is not None:
        utils.save(fig, savename)
    plt.show(fig)
        
folder = '../data/2d_trials'
plot_paths(folder,1,2,np.arange(2))


def plot_this_data(folder,ax,name, cols=[utils.oranges[0],utils.blues[0]],
                   ylab=None, invert=False, leg=True, leg_loc = 'best'):
    if ylab is None:
        ylab=name
    labels = ['PI', 'SLAM']
    data = pd.DataFrame(read(folder))
    ts = data['ts'].values[0]
    n_trials = data.shape[0]
    if name=='dist':
        pi_ydata= np.vstack([np.sqrt(np.sum((data['path'].values[i]-data['pi_sim_path'].values[i])**2,axis=1)) for i in range(n_trials)])
        slam_ydata= np.vstack([np.sqrt(np.sum((data['path'].values[i]-data['slam_sim_path'].values[i])**2,axis=1)) for i in range(n_trials)])
        ax.set_ylim([0,np.max(np.stack([pi_ydata,slam_ydata]))])
    else:
        pi_ydata = np.vstack([d for d in data['pi_' + name].values])
        slam_ydata = np.vstack([d for d in data['slam_' + name].values])
        ax.set_ylim([0,1])
    if invert:
        pi_ydata = 1- pi_ydata
        slam_ydata = 1 - slam_ydata
    for j, ydata in enumerate([pi_ydata, slam_ydata]):
        mean = np.mean(ydata,axis=0)
        lb =np.min(ydata,axis=0)
        ub = np.max(ydata,axis=0)
        ax.fill_between(ts, ub, lb, alpha=.2, color=cols[j])
        ax.plot(ts, mean, color=cols[j], label=labels[j])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylab)
    if leg:
        ax.legend(loc=leg_loc)
    
        


fig = plt.figure(figsize=(7, 3))
ax1 = fig.add_subplot(211)
plot_this_data(folder,ax1,'sim', ylab='Similarity error', invert=True,leg_loc=(0.1,0.6))
ax1 = fig.add_subplot(212)
plot_this_data(folder,ax1,'dist', ylab='Distance', leg=False)

