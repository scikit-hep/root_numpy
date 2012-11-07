#!/usr/bin/env python

from distutils.core import setup, Extension
import distutils.util
import subprocess
import numpy as np
import os
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

module = Extension('root_numpy._librootnumpy',
                   sources=['root_numpy/_librootnumpy.cpp'],
                   include_dirs=[np.get_include(), root_inc, 'root_numpy'],
                   #extra_compile_args = []+root_cflags,
                   extra_link_args=[] + root_ldflags)

setup(name='root_numpy',
       version='1.04',
       description='Convert root tree to numpy array',
       author='Piti Ongmongkolkul',
       author_email='piti118@gmail.com',
       url='https://github.com/piti118/root_numpy',
       package_dir={'root_numpy': 'root_numpy'},
       packages=['root_numpy'],
       ext_modules=[module]
       )
