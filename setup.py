"""
Setup script for the Singularis project.

This script uses setuptools to package the Singularis AGI framework,
defining metadata, dependencies, and other package information. It is used
for installing the project and its required libraries.
"""

from setuptools import setup, find_packages

setup(
    name="singularis",
    version="0.1.0",
    description="The Ultimate Consciousness Engine - Spinozistic AI Architecture",
    author="Singularis Project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "loguru>=0.7.0",
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "aiohttp>=3.9.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
        ],
    },
)
