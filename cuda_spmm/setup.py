from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gnn_eadd_extensions',
    ext_modules=[
        CUDAExtension('custom_spmm', [
            'spmm_kernel.cu',
        ]),
        CUDAExtension('warp_gat', [
            'warp_gat_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
