
============
Installation
============

Requirements
============

* `ROOT <http://root.cern.ch/>`_
* `NumPy <http://numpy.scipy.org/>`_

root_numpy is tested with ROOT 5.32, NumPy 1.6.1, Python 2.7.1 but it should
work in most places.

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

   sudo python setup.py install

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

   sudo pip install root_numpy

Running the Tests
=================

Testing requires the `nose <https://nose.readthedocs.org/en/latest/>`_ package.
Once `root_numpy` is installed, it may be tested (from outside the source
directory) by running::

   nosetests --exe -s -v root_numpy

`root_numpy` can also be tested before installing by running this from inside
the source directory::

   make test

