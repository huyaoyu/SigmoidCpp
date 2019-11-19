from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
      name="SigmoidCppCUDA",
      ext_modules=[
            CUDAExtension("SigmoidCppCUDA", [
                  'SigmoidCpp.cpp',
                  'SigmoidCpp_Kernel.cu',
                  ] )
            ],
      cmdclass={
            'build_ext': BuildExtension
            }
      )