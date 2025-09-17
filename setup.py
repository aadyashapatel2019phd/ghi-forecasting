#!/usr/bin/env python3
"""
Setup script for GHI Forecasting package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
long_description = (Path(__file__).parent / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ghi-forecasting",
    version="1.0.0",
    author="Aadyasha Patel, O. V. Gnana Swathika",
    author_email="your.email@example.com",
    description="Seasonal GHI Forecasting for Stand-Alone Photovoltaic Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aadyashapatel2019phd/ghi-forecasting",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ghi-forecast=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.md"],
    },
    keywords="machine-learning solar-energy ghi-forecasting renewable-energy photovoltaic",
    project_urls={
        "Bug Reports": "https://github.com/aadyashapatel2019phd/ghi-forecasting/issues",
        "Source": "https://github.com/aadyashapatel2019phd/ghi-forecasting",
        "Documentation": "https://github.com/aadyashapatel2019phd/ghi-forecasting#readme",
    },
)