"""
Python package to do thermal modeling on atmosphereless bodies in the Solar system.
"""

from setuptools import setup, find_packages

install_requires = ['numpy',
                    'astropy',
                    'numba']

classifiers = ["Intended Audience :: Science/Research",
               "Operating System :: OS Independent"]

setup(
    name="yssbtmpy",
    version="0.0.1.dev",
    author="Yoonsoo P. Bach",
    author_email="dbstn95@gmail.com",
    description="Python package to do thermal modeling on atmosphereless bodies in the Solar system.",
    license="MIT",
    keywords="",
    url="",
    classifiers=classifiers,
    packages=find_packages(),
    python_requires='>=3.6',
    tests_require=["pytest"],
    setup_requires=["pytest-runner"],
    install_requires=install_requires)
