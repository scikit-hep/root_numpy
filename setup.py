from distutils.core import setup, Extension
import distutils.util
import subprocess
import numpy as np
root_inc = subprocess.check_output(["root-config", "--incdir"]).strip()
root_ldflags = subprocess.check_output(["root-config", "--libs"]).strip().split(' ')
print 'xxx',root_ldflags

print distutils.util.get_platform() 
module = Extension('root_numpy',
                    sources = ['src/root_numpy.cc'],
                    include_dirs= [np.get_include(),root_inc],
                    #extra_compile_args = []+root_cflags,
                    extra_link_args = []+root_ldflags)

setup (name = 'root_numpy',
       version = '1.01',
       description = 'Convert root tree to numpy array',
       author='Piti Ongmongkolkul',
       author_email='piti118@gmail.com',
       url='https://github.com/piti118/root_numpy',
       ext_modules = [module]
       )