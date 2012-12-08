#!/usr/bin/env python

from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages

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
                        stdout=subprocess.PIPE).communicate()[0]
    root_inc = str(root_inc, encoding='utf8').strip()
    root_ldflags = subprocess.Popen(["root-config", "--libs"],
                        stdout=subprocess.PIPE).communicate()[0]
    root_ldflags = str(root_ldflags, encoding='utf8').strip().split()
except OSError:
    rootsys = os.environ['ROOTSYS']
    root_inc = subprocess.Popen([rootsys+"/bin/root-config", "--incdir"],
                        stdout=subprocess.PIPE).communicate()[0]
    root_inc = str(root_inc, encoding='utf8').strip()
    root_ldflags = subprocess.Popen([rootsys+"/bin/root-config", "--libs"],
                        stdout=subprocess.PIPE).communicate()[0]
    root_ldflags = str(root_ldflags, encoding='utf8').strip().split()

module = Extension('root_numpy._librootnumpy',
                   sources=['root_numpy/_librootnumpy.cpp'],
                   include_dirs=[np.get_include(), root_inc, 'root_numpy'],
                   extra_compile_args = [],
                   extra_link_args=[] + root_ldflags)

setup(
    name='root_numpy',
    version='2.00',
    description='ROOT TTree to numpy array converter',
    author='Piti Ongmongkolkul',
    author_email='piti118@gmail.com',
    url='https://github.com/piti118/root_numpy',
    download_url='https://github.com/piti118/root_numpy/archive/v2.00.zip',
    packages=find_packages(),
    package_data={
        'root_numpy': ['tests/*.root']},
    ext_modules=[module],
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
