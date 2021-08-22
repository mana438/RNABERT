from distutils.core import setup, Extension

setup(name = 'alignment_C', version = '1.0.0',  \
   ext_modules = [Extension('alignment_C', ['alignment.cpp'])])