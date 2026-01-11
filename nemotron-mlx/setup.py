"""Setup script for Nemotron Speech ASR MLX."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="nemotron-mlx",
    version="0.1.0",
    description="MLX implementation of NVIDIA Nemotron Speech ASR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MLX Community",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "mlx>=0.30.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "audio": [
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
        ],
        "conversion": [
            "torch>=2.0.0",
            "huggingface_hub>=0.20.0",
            "pyyaml>=6.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
