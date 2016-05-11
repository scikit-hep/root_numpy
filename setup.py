#!/usr/bin/env python

from __future__ import print_function

import sys

# check Python version
if sys.version_info < (2, 6):
    sys.exit("root_numpy only supports python 2.6 and above")

# check that NumPy is installed
try:
    import numpy
except ImportError:
    raise RuntimeError(
        "numpy cannot be imported. numpy must be installed "
        "prior to installing root_numpy")

try:
    # try to use setuptools if installed
    from pkg_resources import parse_version, get_distribution
    from setuptools import setup, Extension
    if get_distribution('setuptools').parsed_version < parse_version('0.7'):
        # before merge with distribute
        raise ImportError
except ImportError:
    # fall back on distutils
    from distutils.core import setup, Extension

import os
from glob import glob

# Prevent setup from trying to create hard links
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

exec(open('root_numpy/setup_utils.py').read())

rootsys = os.getenv('ROOTSYS', None)
if rootsys is not None:
    try:
        root_config = os.path.join(rootsys, 'bin', 'root-config')
        root_version = root_version_installed(root_config)
        root_cflags, root_ldflags = root_flags(root_config)
        has_tmva = root_has_feature('tmva', root_config)
    except OSError:
        raise RuntimeError(
            "ROOTSYS is {0} but running {1} failed".format(
                rootsys, root_config))
else:
    try:
        root_version = root_version_installed()
        root_cflags, root_ldflags = root_flags()
        has_tmva = root_has_feature('tmva')
    except OSError:
        raise RuntimeError(
            "root-config is not in PATH and ROOTSYS is not set. "
            "Is ROOT installed correctly?")

librootnumpy = Extension(
    'root_numpy._librootnumpy',
    sources=[
        'root_numpy/src/_librootnumpy.cpp',
    ],
    depends=glob('root_numpy/src/*.h'),
    language='c++',
    include_dirs=[
        numpy.get_include(),
        'root_numpy/src',
    ],
    extra_compile_args=root_cflags + [
        '-Wno-unused-function',
        '-Wno-write-strings',
    ],
    extra_link_args=root_ldflags + ['-lTreePlayer'])

ext_modules = [librootnumpy]
packages = [
    'root_numpy',
    'root_numpy.testdata',
    'root_numpy.extern',
    ]

if has_tmva:
    librootnumpy_tmva = Extension(
        'root_numpy.tmva._libtmvanumpy',
        sources=[
            'root_numpy/tmva/src/_libtmvanumpy.cpp',
        ],
        depends=['root_numpy/src/2to3.h'],
        language='c++',
        include_dirs=[
            numpy.get_include(),
            'root_numpy/src',
            'root_numpy/tmva/src',
        ],
        extra_compile_args=root_cflags + [
            '-Wno-unused-function',
            '-Wno-write-strings',
        ],
        extra_link_args=root_ldflags + ['-lTMVA'])
    ext_modules.append(librootnumpy_tmva)
    packages.append('root_numpy.tmva')

# check for custom args
filtered_args = []
release = False
install = 'install' in sys.argv
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
        dev_info.replace('.dev0', ''))

exec(open('root_numpy/info.py').read())
if install:
    print(__doc__)

    config = {
        'ROOT_version': root_version,
        'numpy_version': numpy.__version__,
        }

    # write config.json
    import json
    with open('root_numpy/config.json', 'w') as config_file:
        json.dump(config, config_file, indent=4)

setup(
    name='root_numpy',
    version=__version__,
    description='An interface between ROOT and NumPy',
    long_description=''.join(open('README.rst').readlines()[7:]),
    author='the rootpy developers',
    author_email='rootpy-dev@googlegroups.com',
    license='MIT',
    url='http://rootpy.github.io/root_numpy',
    download_url='http://pypi.python.org/packages/source/r/'
                 'root_numpy/root_numpy-{0}.tar.gz'.format(__version__),
    packages=packages,
    package_data={
        'root_numpy': ['testdata/*.root', 'config.json'],
    },
    ext_modules=ext_modules,
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Development Status :: 5 - Production/Stable',
    ]
)

if release:
    # revert root_numpy/info.py
    shutil.move('info.tmp', 'root_numpy/info.py')

if install:
    os.remove('root_numpy/config.json')
