#!/usr/bin/env python3

from setuptools import setup


"""
Setup GraphProt2

NOTE that additional libraries are needed to run GraphProt2. For full 
installation instructions see the README.md at:
https://github.com/BackofenLab/GraphProt2

"""

setup(
    name='graphprot2',
    version='0.2',
    description='Modelling RBP binding preferences to predict RPB binding sites',
    long_description=open('README.md').read(),
    author='Michael Uhl',
    author_email='uhlm@informatik.uni-freiburg.de',
    url='https://github.com/BackofenLab/GraphProt2',
    license='MIT',
    scripts=['bin/graphprot2'],
    packages=['graphprot2'],
    package_data={'graphprot2': ['content/*']},
    zip_safe=False,
)

