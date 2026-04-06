from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_spmm',
    ext_modules=[
        CUDAExtension('custom_spmm', [
            'spmm_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
