#!/usr/bin/env python

import os
import sys
from distutils.core import setup, Extension
import distutils.util
import subprocess
import numpy as np
from glob import glob

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

root_inc = ''
root_ldflags = []
try:
    root_inc = subprocess.Popen(["root-config", "--incdir"],
        stdout=subprocess.PIPE).communicate()[0].strip()
    root_ldflags = subprocess.Popen(["root-config", "--libs"],
        stdout=subprocess.PIPE).communicate()[0].strip().split(' ')
except OSError:
    rootsys = os.environ['ROOTSYS']
    root_inc = subprocess.Popen([rootsys+"/bin/root-config", "--incdir"],
        stdout=subprocess.PIPE).communicate()[0].strip()
    root_ldflags = subprocess.Popen([rootsys+"/bin/root-config", "--libs"],
        stdout=subprocess.PIPE).communicate()[0].strip().split(' ')

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

execfile('root_numpy/info.py')
if 'install' in sys.argv:
    print __doc__

description = open('README.rst').readlines()

setup(
    name='root_numpy',
    version=__version__,
    description='An interface between ROOT and NumPy',
    long_description=''.join(description[2:4] + description[10:]),
    author='the rootpy developers',
    author_email='rootpy-dev@googlegroups.com',
    url='https://github.com/rootpy/root_numpy',
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
