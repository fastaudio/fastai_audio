from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


with open("README.md", "r") as fh:
    long_description = fh.read()


with open('.version') as f:
    version = f.read()

setup(
    name='fastai_audio',
    author="Harry Coultas Blum",
    author_email="harrycblum@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=version,
    packages=find_packages(),
    description='Working with audio and fastaiv1',
    install_requires=requirements,
)
