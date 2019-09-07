from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='fastaiaudio',
    version='0.1',
    packages=find_packages(),
    description='Audio data with fastai',
    install_requires=requirements,
    dependency_links=['git+git://github.com/pytorch/torchaudio@f720aec079'],
)