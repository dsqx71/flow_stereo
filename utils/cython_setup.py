from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess

import numpy

ext_module = Extension(
    "util_cython",
    ["util_cython.pyx"],
)

setup(
	name = 'util_cython',
	cmdclass = {'build_ext': build_ext},
    ext_modules=[ext_module],
    include_dirs=[numpy.get_include()]
)
args = ['cp','./build/lib.linux-x86_64-2.7/util_cython.so','../']
subprocess.call(args)