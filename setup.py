"""


@author: Alex Kerr
"""

from setuptools import find_packages, setup

setup(name="ballnspring",
      version="0.1.0",
      description="A set of routines to calculate motion of a system of \
                  balls and springs analytically.",
      author="Alex Kerr",
      author_email="ajkerr0@gmail.com",
      url="https://github.com/ajkerr0/ballnspring",
      packages=find_packages(),
      install_requires=[
      'numpy', 'scipy',
      ],
      )