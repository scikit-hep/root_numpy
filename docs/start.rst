
===============
Getting Started
===============

Try `root_numpy` on `CERN's LXPLUS <http://information-technology.web.cern.ch/services/lxplus-service>`_
========================================================================================================

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

   >>> from root_numpy import root2rec, testdata
   >>> root2rec(testdata.get_filepath())[:20] # doctest: +SKIP
   rec.array([(1, 1.0, 1.0), (2, 3.0, 4.0), (3, 5.0, 7.0), (4, 7.0, 10.0),
          (5, 9.0, 13.0), (6, 11.0, 16.0), (7, 13.0, 19.0), (8, 15.0, 22.0),
          (9, 17.0, 25.0), (10, 19.0, 28.0), (11, 21.0, 31.0),
          (12, 23.0, 34.0), (13, 25.0, 37.0), (14, 27.0, 40.0),
          (15, 29.0, 43.0), (16, 31.0, 46.0), (17, 33.0, 49.0),
          (18, 35.0, 52.0), (19, 37.0, 55.0), (20, 39.0, 58.0)],
         dtype=[('n_int', '<i4'), ('f_float', '<f4'), ('d_double', '<f8')])


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

