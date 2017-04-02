from numpy.distutils.core import setup, Extension
from os.path import join

extensions = []
src = 'walker/simulators/src'
extensions.append(
    Extension(name = 'walker.simulators.src._diffusion',
              sources = [join(src, "_diffusion.f95")])
)

src = 'walker/simulators/src'
extensions.append(
    Extension(name = 'walker.simulators.src._temp_acc',
              sources = [join(src, "_temp_acc.f95")])
)

setup(name = 'walker',
    version = '0.1',
    description = 'Python API for random walk on Gaussian potential',
    url = 'https://github.com/pgromano/walker',
    packages = ['walker'],
    install_requires=[
        'numpy'
    ],
    ext_modules=extensions,
    zip_safe = False
)
