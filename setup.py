from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_module = Extension(
    "dp",
    ["dp.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
	name = 'dp',
	cmdclass = {'build_ext': build_ext},
    ext_modules=[ext_module],
    include_dirs=[numpy.get_include()]
)    