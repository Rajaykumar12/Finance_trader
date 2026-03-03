"""Build script for the C++ Gamma Exposure engine.

Usage:
    python setup_cpp.py build_ext --inplace
"""

from setuptools import setup, Extension
import pybind11


ext = Extension(
    "gamma_engine",
    sources=["engines/cpp/gamma_engine.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["-std=c++17", "-O3", "-fPIC"],
)

setup(
    name="gamma_engine",
    version="0.1.0",
    ext_modules=[ext],
)
