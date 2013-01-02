#!/usr/bin/env python
from distutils.core import setup, Extension
import distutils.util
import subprocess
import numpy as np
import os
from glob import glob

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
    include_dirs=[
        np.get_include(),
        root_inc,
        'root_numpy/src'],
    extra_compile_args = [],
    extra_link_args=[] + root_ldflags)

libnumpyhist = Extension('root_numpy._libnumpyhist',
    sources=['root_numpy/src/_libnumpyhist.cpp'],
    include_dirs=[np.get_include(), root_inc, 'root_numpy'],
    extra_compile_args = [],
    extra_link_args=[] + root_ldflags)


execfile('root_numpy/info.py')

setup(
    name='root_numpy',
    version=__version__,
    description='An interface between ROOT and NumPy',
    long_description=''.join(open('README.rst').readlines()[2:]),
    author='the rootpy developers',
    author_email='rootpy-dev@googlegroups.com',
    url='https://github.com/rootpy/root_numpy',
    download_url='http://pypi.python.org/packages/source/r/'
                 'root_numpy/root_numpy-%s.tar.gz' % __version__,
    packages=['root_numpy','root_numpy.tests'],
    package_data={
        'root_numpy': ['tests/*.root']},
    ext_modules=[librootnumpy, libnumpyhist],
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
