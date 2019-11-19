from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="SigmoidCpp",
      ext_modules=[cpp_extension.CppExtension("SigmoidCpp", ['SigmoidCpp.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})