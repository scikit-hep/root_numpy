from __future__ import print_function

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
hardware = '{cpu}\nStorage: {hdd}\nROOT-{root}\nPython-{python}\nNumPy-{numpy}'.format(
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
print("{0:>10}  {1:<10}  {2:<10}".format("entries", "root_numpy", "ROOT"))
for entries in num_entries:
    print("{0:>10}".format(entries), end="")
    if entries < 1e3:
        iterations = 200
    elif entries < 1e5:
        iterations = 20
    else:
        iterations = 4
    tree = make_tree(entries, branches=1)
    branchname = tree.GetListOfBranches()[0].GetName()
    root_numpy_times.append(
        min(timeit.Timer('tree2array(tree)',
                         setup='from root_numpy import tree2array; from __main__ import tree').repeat(3, iterations)) / iterations)
    root_times.append(
        min(timeit.Timer('draw("{0}", "", "goff")'.format(branchname),
                         setup='from __main__ import tree; draw = tree.Draw').repeat(3, iterations)) / iterations)
    print("  {0:10.5f}".format(root_numpy_times[-1]), end="")
    print("  {0:10.5f}".format(root_times[-1]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

ax1.plot(num_entries, root_numpy_times, '-o', label='root_numpy.tree2array()', linewidth=1.5)
ax1.plot(num_entries, root_times, '--o', label='ROOT.TTree.Draw()', linewidth=1.5)
ax1.set_xscale("log", nonposx='clip')
ax1.set_yscale("log", nonposx='clip')
ax1.legend(loc=(0.03, 0.7), frameon=False, fontsize=10)
ax1.set_ylabel('time [s]')
ax1.set_xlabel('number of entries')
ax1.text(0.03, 0.97, 'tree contains a single branch',
         verticalalignment='top', horizontalalignment='left',
         transform=ax1.transAxes, fontsize=12)

# time vs branches
num_branches = np.linspace(1, 10, 10, dtype=np.int)
root_numpy_times = []
root_times = []
print("\n{0:>10}  {1:<10}  {2:<10}".format("branches", "root_numpy", "ROOT"))
for branches in num_branches:
    print("{0:>10}".format(branches), end="")
    tree = make_tree(1000000, branches=branches)
    branchnames = [branch.GetName() for branch in tree.GetListOfBranches()]
    branchname = ':'.join(branchnames)
    iterations = 5
    root_numpy_times.append(
        min(timeit.Timer('tree2array(tree)',
                         setup='from root_numpy import tree2array; from __main__ import tree').repeat(3, iterations)) / iterations)
    root_times.append(
        min(timeit.Timer('draw("{0}", "", "goff candle")'.format(branchname),
                         setup='from __main__ import tree; draw = tree.Draw').repeat(3, iterations)) / iterations)
    print("  {0:10.5f}".format(root_numpy_times[-1]), end="")
    print("  {0:10.5f}".format(root_times[-1]))

ax2.plot(num_branches, root_numpy_times, '-o', label='root_numpy.tree2array()', linewidth=1.5)
ax2.plot(num_branches, root_times, '--o', label='ROOT.TTree.Draw()', linewidth=1.5)
#ax2.legend(loc='lower right', frameon=False, fontsize=12)
ax2.set_ylabel('time [s]')
ax2.set_xlabel('number of branches')
ax2.text(0.03, 0.97, 'tree contains 1M entries per branch',
         verticalalignment='top', horizontalalignment='left',
         transform=ax2.transAxes, fontsize=12)
ax2.text(0.03, 0.87, hardware,
         verticalalignment='top', horizontalalignment='left',
         transform=ax2.transAxes, fontsize=10)

fig.tight_layout()
fname = 'bench_tree2array_{0}.{1}'
ipng = 0
while os.path.exists(fname.format(ipng, 'png')):
    ipng += 1
fig.savefig(fname.format(ipng, 'png'), transparent=True)
fig.savefig(fname.format(ipng, 'pdf'), transparent=True)
