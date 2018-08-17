from setuptools import setup, find_packages

setup(
    name='ContraQA',
    version='0.1',
    description='Workbench for dialog agents training and evaluation',
    author='Felipe Salvatore',
    packages=find_packages(),
    test_suite="tests"
)
