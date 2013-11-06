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

* ROOT 5.32, 5.34
* NumPy 1.6, 1.7, 1.8
* Python 2.6, 2.7
* GNU/Linux, Mac OS

.. warning:: **Mac OS:** if you're compiling root_numpy with Clang and
   linking against libc++, you will also need to compile ROOT with Clang and
   libc++, because libstdc++ and libc++ are not ABI compatible.

   ROOT compiles with Clang and libc++ since version 5.34/11, but PyROOT has a
   bug which was fixed after that, so it is best to compile from the
   v5-34-00-patches branch of ROOT. You can do this easily with Homebrew via::

      brew install --HEAD root

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

.. note:: This will install the latest version of root_numpy on PyPI which may
   be lacking many new unreleased features.

To install in your `home directory
<http://www.python.org/dev/peps/pep-0370/>`_::

   pip install --user root_numpy

To install system-wide (requires root privileges)::

   sudo ROOTSYS=$ROOTSYS pip install root_numpy

Running the Tests
=================

Testing requires the `nose <https://nose.readthedocs.org/en/latest/>`_ package.
Once `root_numpy` is installed, it may be tested (from outside the source
directory) by running::

   nosetests --exe -s -v root_numpy

`root_numpy` can also be tested before installing by running this from inside
the source directory::

   make test

