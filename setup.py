"""
setup.py — Installation configuration for AGE (Autonomous Geometric Engine)
"""
from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

setup(
    name="autonomous-geometric-engine",
    version="1.0.0",
    author="AGE Contributors",
    author_email="contact@age-project.org",
    description="A production-ready clustering estimator with out-of-sample prediction and OOD rejection",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/autonomous-geometric-engine",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "hdbscan": ["hdbscan>=0.8.0"],
    },
    entry_points={
        "console_scripts": [
            "age-benchmark=benchmark_age:main",
        ],
    },
    keywords="clustering, manifold-learning, ood-detection, out-of-sample, geometric, ensemble",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/autonomous-geometric-engine/issues",
        "Source": "https://github.com/yourusername/autonomous-geometric-engine",
        "Documentation": "https://github.com/yourusername/autonomous-geometric-engine/blob/main/README.md",
    },
)