"""Installation script for the 'test_legged_gym' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "isaacgym",
    "rsl_rl",
]

# Installation operation
setup(
    name="test_legged_gym",
    version="1.0.0",
    author="Yuxuan Xie",
    packages=find_packages(),
    author_email="yuxuangmxie@gmail.com",
    description="Isaac Gym environments  for legged robots",
    install_requires=INSTALL_REQUIRES,
)