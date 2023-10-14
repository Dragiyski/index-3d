from setuptools import setup, Extension
import numpy

setup(
    ext_modules=[
        Extension(
            name='dragiyski.i3d.floatlist',
            sources=['src/dragiyski/i3d/floatlist/main.cxx'],
            include_dirs=[
                numpy.get_include()
            ],
            swig_opts=['-O0']
        )
    ]
)