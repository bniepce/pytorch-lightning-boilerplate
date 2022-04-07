from setuptools import setup, find_packages

setup(
    name='pytorch-boilerplate',
    version='0.0.0',
    description='A PyTorch boilerplate based on Pytorch Lightning',
    author='Brad Niepceron',
    url='https://github.com/bniepce/pytorch-lightning-boilerplate',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)