from rootpy.io import TemporaryFile
import rootpy
from root_numpy import array2tree
import numpy as np
import uuid
import random
import string
import timeit
import pickle
import platform
import matplotlib.pyplot as plt
import os

with open('hardware.pkl', 'r') as pkl:
    info = pickle.load(pkl)

# construct system hardware information string
hardware = '{cpu}\nStorage: {hdd}\nROOT-{root} Python-{python} NumPy-{numpy}'.format(
    cpu=info['CPU'], hdd=info['HDD'],
    root=rootpy.ROOT_VERSION, python=platform.python_version(),
    numpy=np.__version__)

rfile = TemporaryFile()

def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))

def make_tree(entries, branches=10, dtype=np.double):
    dtype = np.dtype([(randomword(20), dtype) for idx in range(branches)])
    array = np.zeros(entries, dtype=dtype)
    return array2tree(array, name=uuid.uuid4().hex)

# time vs entries
num_entries = np.logspace(1, 7, 20, dtype=np.int)
root_numpy_times = []
root_times = []
for entries in num_entries:
    print(entries)
    iterations = 20 if entries < 1e5 else 4
    tree = make_tree(entries, branches=1)
    branchname = tree.GetListOfBranches()[0].GetName()
    root_numpy_times.append(
        min(timeit.Timer('tree2array(tree)',
                         setup='from root_numpy import tree2array; from __main__ import tree').repeat(3, iterations)) / iterations)
    root_times.append(
        min(timeit.Timer('draw("{0}", "", "goff")'.format(branchname),
                         setup='from __main__ import tree; draw = tree.Draw').repeat(3, iterations)) / iterations)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(num_entries, root_numpy_times, '-o', label='root_numpy.tree2array()', linewidth=1.5)
ax1.plot(num_entries, root_times, '--o', label='ROOT.TTree.Draw()', linewidth=1.5)
ax1.set_xscale("log", nonposx='clip')
ax1.set_yscale("log", nonposx='clip')
ax1.legend(loc='lower right', frameon=False, fontsize=12)
ax1.set_ylabel('time [s]')
ax1.set_xlabel('number of entries')
ax1.text(0.05, 0.95, 'tree contains a single branch',
         verticalalignment='top', horizontalalignment='left',
         transform=ax1.transAxes, fontsize=12)
ax1.text(0.05, 0.85, hardware,
         verticalalignment='top', horizontalalignment='left',
         transform=ax1.transAxes, fontsize=10)

# time vs branches
num_branches = np.linspace(1, 10, 10, dtype=np.int)
root_numpy_times = []
root_times = []
for branches in num_branches:
    print(branches)
    tree = make_tree(1000000, branches=branches)
    branchnames = [branch.GetName() for branch in tree.GetListOfBranches()]
    branchname = ':'.join(branchnames)
    root_numpy_times.append(
        min(timeit.Timer('tree2array(tree)',
                         setup='from root_numpy import tree2array; from __main__ import tree').repeat(3, 3)) / 3)
    root_times.append(
        min(timeit.Timer('draw("{0}", "", "goff candle")'.format(branchname),
                         setup='from __main__ import tree; draw = tree.Draw').repeat(3, 3)) / 3)

ax2.plot(num_branches, root_numpy_times, '-o', label='root_numpy.tree2array()', linewidth=1.5)
ax2.plot(num_branches, root_times, '--o', label='ROOT.TTree.Draw()', linewidth=1.5)
ax2.legend(loc='lower right', frameon=False, fontsize=12)
ax2.set_ylabel('time [s]')
ax2.set_xlabel('number of branches')
ax2.text(0.05, 0.95, 'tree contains 1M entries per branch',
         verticalalignment='top', horizontalalignment='left',
         transform=ax2.transAxes, fontsize=12)

fig.tight_layout()
fname = 'bench_tree2array_{0}.png'
ipng = 0
while os.path.exists(fname.format(ipng)):
    ipng += 1
fig.savefig(fname.format(ipng), transparent=True)
