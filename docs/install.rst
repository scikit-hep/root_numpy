.. include:: references.txt

============
Installation
============

Requirements
============

root_numpy requires that both ROOT_ and NumPy_ are installed
and that ROOT is setup with the ``$ROOTSYS`` environment variable set before
installation. root_numpy's installation script depends on ROOT's
``root-config`` utility to determine the ROOT compiler/linker flags
and include paths. The installation attempts to run ``root-config`` and if
unsuccessful then ``$ROOTSYS/bin/root-config``. If ``root-config`` still cannot
be found, the installation aborts.

root_numpy has been tested with:

* ROOT 5.32, 5.34, 6.04, 6.06, 6.09
* NumPy 1.6, 1.7, 1.8, 1.9, 1.10, 1.11
* Python 2.6, 2.7, 3.4, 3.5
* GNU/Linux, Mac OS

.. warning:: **Mac OS:** libstdc++ and libc++ are not ABI-compatible.

   If you're compiling root_numpy with Clang and linking against libc++, ROOT
   should also have been compiled with Clang and libc++. ROOT compiles with
   Clang and libc++ since version 5.34/11, but PyROOT had a bug that was fixed
   after that tag, so it is best to compile a newer version. You can do this
   easily with Homebrew via::

      brew install --HEAD root

   This issue also comes up if you're using a Python bundle such as Anaconda
   or Enthought Canopy. These bundles build against libstdc++, the GCC C++
   standard library, which is used on Mac OS 10.5 and later. On Mac OS 10.7 and
   later, however, the default compiler links against libc++, the Clang C++
   standard library.


Getting the Latest Source
=========================

Clone the repository with git::

   git clone git://github.com/rootpy/root_numpy.git

or checkout with svn::

   svn checkout https://github.com/rootpy/root_numpy/trunk root_numpy

Manual Installation
===================

If you have obtained a copy of `rootpy_numpy` yourself use the ``setup.py``
script to install.

To install in your `home directory
<http://www.python.org/dev/peps/pep-0370/>`_::

   python setup.py install --user

To install system-wide (requires root privileges)::

   sudo ROOTSYS=$ROOTSYS python setup.py install

Automatic Installation
======================

To install a `released version
<http://pypi.python.org/pypi/root_numpy/>`_ of
`root_numpy` use `pip <http://pypi.python.org/pypi/pip>`_.


To install in your `home directory
<http://www.python.org/dev/peps/pep-0370/>`_::

   pip install --user root_numpy

To install system-wide (requires root privileges)::

   sudo ROOTSYS=$ROOTSYS pip install root_numpy

.. note:: The above will install the latest version of root_numpy on PyPI
   and may be lacking new unreleased features. You can also use pip to
   install the latest version of root_numpy on github::

       pip install --upgrade --user https://github.com/rootpy/root_numpy/zipball/master

Running the Tests
=================

Testing requires the `nose <https://nose.readthedocs.org/en/latest/>`_ package.
Once `root_numpy` is installed, it may be tested (from outside the source
directory) by running::

   nosetests --exe -s -v root_numpy

`root_numpy` can also be tested before installing by running this from inside
the source directory::

   make test

