#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='voltaml',
    version='0.1.0',
    description='VoltaML',
    author='Abhiroop Tejomay',
    author_email='abhirooptejomay@gmail.com',
    url='https://github.com/visualCalculus/apache-tvm-wrapper',
    packages=find_packages(include=["voltaml", "voltaml.*"]),
)