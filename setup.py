#!/usr/bin/env python

from __future__ import print_function

import sys

# Check Python version
if sys.version_info < (2, 6):
    sys.exit("root_numpy only supports python 2.6 and above")

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

try:
    # Try to use setuptools if installed
    from setuptools import setup, Extension
    from pkg_resources import parse_version, get_distribution

    if get_distribution('setuptools').parsed_version < parse_version('0.7'):
        # setuptools is too old (before merge with distribute)
        raise ImportError

    from setuptools.command.build_ext import build_ext as _build_ext
    from setuptools.command.install import install as _install
    use_setuptools = True

except ImportError:
    # Use distutils instead
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext as _build_ext
    from distutils.command.install import install as _install
    use_setuptools = False

import os
from glob import glob
from contextlib import contextmanager

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

@contextmanager
def version(release=False):
    if not release:
        yield
    else:
        # Remove dev from version in root_numpy/info.py
        import shutil
        print("writing release version in 'root_numpy/info.py'")
        shutil.move('root_numpy/info.py', 'info.tmp')
        dev_info = ''.join(open('info.tmp', 'r').readlines())
        open('root_numpy/info.py', 'w').write(
            dev_info.replace('.dev0', ''))
        try:
            yield
        finally:
            # Revert root_numpy/info.py
            print("restoring dev version in 'root_numpy/info.py'")
            shutil.move('info.tmp', 'root_numpy/info.py')


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process
        try:
            del builtins.__NUMPY_SETUP__
        except AttributeError:
            pass
        import numpy
        self.include_dirs.append(numpy.get_include())


class install(_install):
    def run(self):
        print(__doc__)
        import numpy

        config = {
            'ROOT_version': str(root_version),
            'numpy_version': numpy.__version__,
        }

        # Write version info in config.json
        print("writing 'root_numpy/config.json'")
        import json
        with open('root_numpy/config.json', 'w') as config_file:
            json.dump(config, config_file, indent=4)

        _install.run(self)

        print("removing 'root_numpy/config.json'")
        os.remove('root_numpy/config.json')


librootnumpy = Extension(
    'root_numpy._librootnumpy',
    sources=[
        'root_numpy/src/_librootnumpy.cpp',
    ],
    depends=glob('root_numpy/src/*.h'),
    language='c++',
    include_dirs=[
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
        depends=glob('root_numpy/tmva/src/*.h') + [
            'root_numpy/src/2to3.h',
        ],
        language='c++',
        include_dirs=[
            'root_numpy/src',
            'root_numpy/tmva/src',
        ],
        define_macros=[('NEW_TMVA_API', None)] if root_version >= '6.07/04' else [],
        extra_compile_args=root_cflags + [
            '-Wno-unused-function',
            '-Wno-write-strings',
        ],
        extra_link_args=root_ldflags + ['-lTMVA'])
    ext_modules.append(librootnumpy_tmva)
    packages.append('root_numpy.tmva')


def setup_package():
    # Only add numpy to *_requires lists if not already installed to prevent
    # pip from trying to upgrade an existing numpy and failing.
    try:
        import numpy
    except ImportError:
        build_requires = ['numpy']
    else:
        build_requires = []

    if use_setuptools:
        setuptools_options = dict(
            setup_requires=build_requires,
            install_requires=build_requires,
            extras_require={
                'with-numpy': ('numpy',),
            },
            zip_safe=False,
        )
    else:
        setuptools_options = dict()

    setup(
        name='root_numpy',
        version=__version__,
        description='The interface between ROOT and NumPy',
        long_description=''.join(open('README.rst').readlines()[7:-4]),
        author='the root_numpy developers',
        author_email='rootpy-dev@googlegroups.com',
        maintainer='Noel Dawe',
        maintainer_email='noel@dawe.me',
        license='new BSD',
        url='http://rootpy.github.io/root_numpy',
        download_url='http://pypi.python.org/packages/source/r/'
                    'root_numpy/root_numpy-{0}.tar.gz'.format(__version__),
        packages=packages,
        package_data={
            'root_numpy': ['testdata/*.root', 'config.json'],
        },
        ext_modules=ext_modules,
        cmdclass={
            'build_ext': build_ext,
            'install': install,
        },
        classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Topic :: Utilities',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: C++',
            'Programming Language :: Cython',
            'Development Status :: 5 - Production/Stable',
        ],
        **setuptools_options
    )


with version(release=set(['sdist', 'register']).intersection(sys.argv[1:])):
    exec(open('root_numpy/info.py').read())
    setup_package()
