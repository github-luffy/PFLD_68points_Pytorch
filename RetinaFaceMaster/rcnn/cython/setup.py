# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["bbox.pyx", "anchors.pyx", "cpu_nms.pyx"]),
    include_dirs=[numpy.get_include()]
)