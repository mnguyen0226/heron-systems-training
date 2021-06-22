# Copyright (C) 2021 Heron Systems, Inc.
from setuptools import find_packages
from setuptools import setup


def long_description():
    with open("README.md", "r") as fh:
        return fh.read()


setup(
    name="gus",
    version="0.0.1dev0",
    author="Karthik Narayan",
    author_email="karthik@starfruit-llc.com",
    description=(
        "A library which samples lattice points uniformly at random from the polytope defined by "
        "the region Ax <= b, x >= 0 where A and b only consist of non-negative, rational numbers."
    ),
    long_description=long_description(),
    long_description_content_type="text/markdown",
    url="",
    license="Proprietary and Confidential",
    python_requires=">=3.7.0",
    packages=find_packages(),
    include_package_data=True,
)
