#!/usr/bin/env python

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "numpy cannot be imported. numpy must be installed "
        "prior to installing root_numpy")

import os
import sys
import subprocess
from glob import glob

from distutils.core import setup, Extension
import distutils.util

# Prevent distutils from trying to create hard links
# which are not allowed on AFS between directories.
# This is a hack to force copying.
try:
    del os.link
except AttributeError:
    pass

local_path = os.path.dirname(os.path.abspath(__file__))
# setup.py can be called from outside the root_numpy directory
os.chdir(local_path)
sys.path.insert(0, local_path)


def root_flags(root_config='root-config'):
    root_inc = subprocess.Popen([root_config, '--incdir'],
        stdout=subprocess.PIPE).communicate()[0].strip()
    root_ldflags = subprocess.Popen([root_config, '--libs'],
        stdout=subprocess.PIPE).communicate()[0].strip().split(' ')
    return root_inc, root_ldflags

try:
    root_inc, root_ldflags = root_flags()
except OSError:
    rootsys = os.getenv('ROOTSYS', None)
    if rootsys is None:
        raise RuntimeError(
            "root-config is not in PATH and ROOTSYS is not set. "
            "Is ROOT installed and setup properly?")
    try:
        root_config = os.path.join(rootsys, 'bin', 'root-config')
        root_inc, root_ldflags = root_flags(root_config)
    except OSError:
        raise RuntimeError(
            "ROOTSYS is {0} but running {1} failed".format(
                rootsys, root_config))

librootnumpy = Extension('root_numpy._librootnumpy',
    sources=['root_numpy/src/_librootnumpy.cpp'],
    language='c++',
    include_dirs=[
        np.get_include(),
        root_inc,
        'root_numpy/src'],
    extra_compile_args = [],
    extra_link_args=[] + root_ldflags + ['-lTreePlayer'])

libnumpyhist = Extension('root_numpy._libnumpyhist',
    sources=['root_numpy/src/_libnumpyhist.cpp'],
    include_dirs=[np.get_include(), root_inc, 'root_numpy'],
    extra_compile_args = [],
    extra_link_args=[] + root_ldflags)

libinnerjoin = Extension('root_numpy._libinnerjoin',
    sources=['root_numpy/src/_libinnerjoin.cpp'],
    include_dirs=[np.get_include(), 'root_numpy'],
    extra_compile_args = [],
    extra_link_args=[])

# check for custom args
filtered_args = []
release = False
for arg in sys.argv:
    if arg == '--release':
        # --release sets the version number before installing
        release = True
    else:
        filtered_args.append(arg)
sys.argv = filtered_args

if release:
    # remove dev from version in root_numpy/info.py
    import shutil
    shutil.move('root_numpy/info.py', 'info.tmp')
    dev_info = ''.join(open('info.tmp', 'r').readlines())
    open('root_numpy/info.py', 'w').write(
        dev_info.replace('.dev', ''))

execfile('root_numpy/info.py')
if 'install' in sys.argv:
    print __doc__

setup(
    name='root_numpy',
    version=__version__,
    description='An interface between ROOT and NumPy',
    long_description=''.join(open('README.rst').readlines()[7:]),
    author='the rootpy developers',
    author_email='rootpy-dev@googlegroups.com',
    url='http://rootpy.github.io/root_numpy',
    download_url='http://pypi.python.org/packages/source/r/'
                 'root_numpy/root_numpy-%s.tar.gz' % __version__,
    packages=[
        'root_numpy',
        'root_numpy.testdata',
        'root_numpy.extern',
    ],
    package_data={
        'root_numpy': ['testdata/*.root'],
    },
    ext_modules=[
        librootnumpy,
        libnumpyhist,
        libinnerjoin,
    ],
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Utilities",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
    ]
)

if release:
    # revert root_numpy/info.py
    shutil.move('info.tmp', 'root_numpy/info.py')
