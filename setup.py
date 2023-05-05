from setuptools import setup, find_packages

setup(
    name="pycompete",
    description="Tools to support the training and evalaution of models for the CRUNCHDAO project.",
    long_description="Tools to support the training and evalaution of models for the CRUNCHDAO project.",
    long_description_content_type='text/markdown',
    author="Francis P. Chmiel",
    url='https://github.com/FChmiel/pycompete',
    license='MIT',
    packages=find_packages(where="pycompete"),  # Required,
)