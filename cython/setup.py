from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
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