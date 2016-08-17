
===============
Getting Started
===============

If you have access to CERN's CVMFS then you can activate an environment with
compatible builds of python, ROOT, numpy and root_numpy with the following::

   export LCGENV_PATH=/cvmfs/sft.cern.ch/lcg/releases
   /cvmfs/sft.cern.ch/lcg/releases/lcgenv/latest/lcgenv -p LCG_85swan2 --ignore Grid x86_64-slc6-gcc49-opt root_numpy > lcgenv.sh
   echo 'export PATH=$HOME/.local/bin:$PATH' >> lcgenv.sh
   source lcgenv.sh

In new terminal sessions, only the last line above will be required.

If you want to instead use your own installation of root_numpy along with any
other packages you need, then continue with setting up a virtualenv. First
install pip and virtualenv::

   curl -O https://bootstrap.pypa.io/get-pip.py
   python get-pip.py --user
   pip install --user virtualenv

Then create and activate a virtualenv (change ``my_env`` at your will)::

   virtualenv my_env
   source my_env/bin/activate

Now install NumPy and root_numpy::

   pip install numpy
   pip install root_numpy

Note that neither ``sudo`` nor ``--user`` is used, because we are in a
virtualenv.

root_numpy should now be ready to use::

   >>> from root_numpy import root2array, testdata
   >>> root2array(testdata.get_filepath('single1.root'))[:20] # doctest: +SKIP
   rec.array([(1, 1.0, 1.0), (2, 3.0, 4.0), (3, 5.0, 7.0), (4, 7.0, 10.0),
          (5, 9.0, 13.0), (6, 11.0, 16.0), (7, 13.0, 19.0), (8, 15.0, 22.0),
          (9, 17.0, 25.0), (10, 19.0, 28.0), (11, 21.0, 31.0),
          (12, 23.0, 34.0), (13, 25.0, 37.0), (14, 27.0, 40.0),
          (15, 29.0, 43.0), (16, 31.0, 46.0), (17, 33.0, 49.0),
          (18, 35.0, 52.0), (19, 37.0, 55.0), (20, 39.0, 58.0)],
         dtype=[('n_int', '<i4'), ('f_float', '<f4'), ('d_double', '<f8')])


A Quick Tutorial
================

For example, get a structured NumPy array from a TTree (copy and paste the
following examples into your Python prompt):

.. code-block:: python

   from root_numpy import root2array, tree2array
   from root_numpy.testdata import get_filepath

   filename = get_filepath('test.root')

   # Convert a TTree in a ROOT file into a NumPy structured array
   arr = root2array(filename, 'tree')
   # The TTree name is always optional if there is only one TTree in the file

   # Or first get the TTree from the ROOT file
   import ROOT
   rfile = ROOT.TFile(filename)
   intree = rfile.Get('tree')

   # and convert the TTree into an array
   array = tree2array(intree)

Include specific branches or expressions and only entries passing a selection:

.. code-block:: python

   array = tree2array(intree,
       branches=['x', 'y', 'sqrt(y)', 'TMath::Landau(x)', 'cos(x)*sin(y)'],
       selection='z > 0',
       start=0, stop=10, step=2)

The above conversion creates an array with five columns from the branches
x and y where z is greater than zero and only looping on the first ten entries
in the tree while skipping every second entry.

Now convert our array back into a TTree:

.. code-block:: python

   from root_numpy import array2tree, array2root

   # Rename the fields
   array.dtype.names = ('x', 'y', 'sqrt_y', 'landau_x', 'cos_x_sin_y')

   # Convert the NumPy array into a TTree
   tree = array2tree(array, name='tree')

   # Or write directly into a ROOT file without using PyROOT
   array2root(array, 'selected_tree.root', 'tree')

root_numpy also provides a function for filling a ROOT histogram from a NumPy
array:

.. code-block:: python

   from ROOT import TH2D
   from root_numpy import fill_hist
   import numpy as np

   # Fill a ROOT histogram from a NumPy array
   hist = TH2D('name', 'title', 20, -3, 3, 20, -3, 3)
   fill_hist(hist, np.random.randn(1000000, 2))
   hist.Draw('LEGO2')

and a function for creating a random NumPy array by sampling a ROOT function
or histogram:

.. code-block:: python

   from ROOT import TF2, TH1D
   from root_numpy import random_sample

   # Sample a ROOT function
   func = TF2('func', 'sin(x)*sin(y)/(x*y)')
   arr = random_sample(func, 1000000)

   # Sample a ROOT histogram
   hist = TH1D('hist', 'hist', 10, -3, 3)
   hist.FillRandom('gaus')
   arr = random_sample(hist, 1000000)


Have Questions or Found a Bug?
==============================

Think you found a bug? Open a new issue here:
`github.com/rootpy/root_numpy/issues <https://github.com/rootpy/root_numpy/issues>`_.

Also feel free to post questions or follow discussion on the
`rootpy-users <http://groups.google.com/group/rootpy-users>`_ or
`rootpy-dev <http://groups.google.com/group/rootpy-dev>`_ Google groups.


Contributing
============

Please post on the rootpy-dev@googlegroups.com list if you have ideas
or contributions. Feel free to fork
`root_numpy on GitHub <https://github.com/rootpy/root_numpy>`_
and later submit a pull request.

