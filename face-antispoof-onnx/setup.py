"""Setup script for face-antispoof-onnx package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding="utf-8")
install_requires = [
    line.strip()
    for line in requirements.split("\n")
    if line.strip() and not line.startswith("#")
]

setup(
    name="face-antispoof-onnx",
    version="1.0.0",
    author="Face Liveness Contributors",
    author_email="",
    description="Production-ready face anti-spoofing with ONNX runtime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/face-antispoof-onnx",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "face-antispoof=src.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
