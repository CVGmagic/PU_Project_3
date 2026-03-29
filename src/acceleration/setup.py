from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Provides setup for barnes_hut.cpp
# I have no Idea how this works, please dont change anything

ext_modules = [
    Pybind11Extension(
        "barnes_hut",          # name of the module (import barnes_hut)
        ["barnes_hut.cpp"],  # your cpp file
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)