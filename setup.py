from setuptools import setup

setup(name = 'walker',
    version = '0.1',
    description = 'Python API for random walk on Gaussian potential',
    url = 'https://github.com/pgromano/walker',
    packages = ['walker'],
    install_requires=[
        'numpy',
        'tqdm',
        'theano'
    ],
    zip_safe = False
)
