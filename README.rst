.. -*- mode: rst -*-

`[see full documentation] <http://rootpy.github.com/root_numpy/>`_

.. |ci| image:: https://travis-ci.org/rootpy/root_numpy.png
   :target: https://travis-ci.org/rootpy/root_numpy

root_numpy |ci|
===============

root_numpy is a Python extension for converting
`ROOT TTrees <http://root.cern.ch/root/html/TTree.html>`_ into NumPy
`recarrays <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html>`_
or `structured arrays <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
as well as converting NumPy arrays back into ROOT TTrees.
With the core internals written in C++, root_numpy can efficiently handle large
amounts of data (limited only by the available memory).
Now that your ROOT data is in NumPy form, you can make use of the many powerful
scientific Python packages or perform quick exploratory data analysis in
interactive environments like `IPython <http://ipython.org/>`_ (especially
IPython's popular `notebook <http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html>`_
feature).

root_numpy currently supports basic types such as bool, int, float,
double, etc. and arrays of basic types (both variable and fixed-length).
Vectors of basic types are also supported.

See the root2hdf5 script in the `rootpy <https://github.com/rootpy/rootpy>`_
package that uses root_numpy and `PyTables <http://www.pytables.org>`_ to
convert all TTrees in a ROOT file into the
`HDF5 <http://www.hdfgroup.org/HDF5/>`_ format.

Requirements
------------

* `ROOT <http://root.cern.ch/>`_
* `NumPy <http://numpy.scipy.org/>`_

root_numpy is tested with ROOT 5.32, NumPy 1.6.1, Python 2.7.1 but it should
work in most places.

Typical Usage
-------------

.. code-block:: python

   >>> import ROOT
   >>> from root_numpy import root2array, root2rec, tree2rec
   >>> # convert into a numpy structured array
   >>> # treename is always optional if there is only one tree in the file
   >>> arr = root2array('a.root', 'treename')
   >>> # convert into a numpy record array
   >>> rec = root2rec('a.root', 'treename')
   >>> # or directly convert a tree
   >>> rfile = ROOT.TFile('a.root')
   >>> tree = rfile.Get('treename')
   >>> rec = tree2rec(tree)

Getting the Latest Source
-------------------------

Clone the repository with git::

   git clone git://github.com/rootpy/root_numpy.git

or checkout with svn::

   svn checkout http://svn.github.com/rootpy/root_numpy

Manual Installation
-------------------

If you have obtained a copy of `rootpy_numpy` yourself use the ``setup.py``
script to install.

To install in your `home directory
<http://www.python.org/dev/peps/pep-0370/>`_::

   python setup.py install --user

To install system-wide (requires root privileges)::

   sudo python setup.py install

Automatic Installation
----------------------

To install a `released version
<http://pypi.python.org/pypi/root_numpy/>`_ of
`root_numpy` use `pip <http://pypi.python.org/pypi/pip>`_.

.. note:: This will install the latest version of root_numpy on PyPI which may
   be lacking many new unreleased features.

To install in your `home directory
<http://www.python.org/dev/peps/pep-0370/>`_::

   pip install --user root_numpy

To install system-wide (requires root privileges)::

   sudo pip install root_numpy

Try `root_numpy` on `CERN's LXPLUS <http://information-technology.web.cern.ch/services/lxplus-service>`_
--------------------------------------------------------------------------------------------------------

First, `set up ROOT <http://root.cern.ch/drupal/content/starting-root>`_::

   source /afs/cern.ch/sw/lcg/contrib/gcc/4.3/x86_64-slc5/setup.sh &&\
   cd /afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.00/x86_64-slc5-gcc43-opt/root &&\
   source bin/thisroot.sh &&\
   cd -

Then, create and activate a `virtualenv <https://pypi.python.org/pypi/virtualenv>`_ (change `my_env` at your will)::

   virtualenv my_env # necessary only the first time
   source my_env/bin/activate

Install NumPy::

   pip install numpy

Get the `latest source <https://github.com/rootpy/root_numpy#getting-the-latest-source>`_::

   git clone https://github.com/rootpy/root_numpy.git

and `install <https://github.com/rootpy/root_numpy#manual-installation>`_ it::

   ~/my_env/bin/python root_numpy/setup.py install

Note that neither `sudo` nor `--user` is used, because we are in a virtualenv.

`root_numpy` should now be ready to `use <http://rootpy.github.com/root_numpy/>`_:

.. code-block:: python

   >>> from root_numpy import testdata, root2rec
   >>> root2rec(testdata.get_filepath())[:20]
   rec.array([(1, 1.0, 1.0), (2, 3.0, 4.0), (3, 5.0, 7.0), (4, 7.0, 10.0),
         (5, 9.0, 13.0), (6, 11.0, 16.0), (7, 13.0, 19.0), (8, 15.0, 22.0),
         (9, 17.0, 25.0), (10, 19.0, 28.0), (11, 21.0, 31.0),
         (12, 23.0, 34.0), (13, 25.0, 37.0), (14, 27.0, 40.0),
         (15, 29.0, 43.0), (16, 31.0, 46.0), (17, 33.0, 49.0),
         (18, 35.0, 52.0), (19, 37.0, 55.0), (20, 39.0, 58.0)],
         dtype=[('n_int', '<i4'), ('f_float', '<f4'), ('d_double', '<f8')])

Running the Tests
-----------------

Testing requires the `nose <https://nose.readthedocs.org/en/latest/>`_ package.
Once `root_numpy` is installed, it may be tested (from outside the source
directory) by running::

   nosetests --exe -s -v root_numpy

`root_numpy` can also be tested before installing by running this from inside
the source directory::

   make test

Development
-----------

Please post on the rootpy-dev@googlegroups.com list if you have ideas
or contributions. Feel free to fork
`root_numpy on GitHub <https://github.com/rootpy/root_numpy>`_
and later submit a pull request.

Have Questions or Found a Bug?
------------------------------

Think you found a bug? Open a new issue here:
`github.com/rootpy/root_numpy/issues <https://github.com/rootpy/root_numpy/issues>`_.

Also feel free to post questions or follow discussion on the
`rootpy-users <http://groups.google.com/group/rootpy-users>`_ or
`rootpy-dev <http://groups.google.com/group/rootpy-dev>`_ Google groups.
