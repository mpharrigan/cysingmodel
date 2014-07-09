from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(
        name='Cysingmodel',
        ext_modules = cythonize("cysingmodel/_ising.pyx")
)
