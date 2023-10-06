from setuptools import setup, Extension
import numpy

setup(
    ext_modules=[
        Extension(
            name='floatlist',
            sources=['src/floatlist/main.cxx'],
            include_dirs=[
                numpy.get_include()
            ],
            swig_opts=['-O0']
        )
    ]
)