#!/usr/bin/env python3
"""
BabyBIONN Setup Configuration for PyTorch
"""
import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.extension import Extension

# Check Python version
if sys.version_info < (3, 8):
    print("BabyBIONN requires Python 3.8 or higher")
    sys.exit(1)

def read_requirements():
    """Read requirements from requirements.txt"""
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        with open(req_path, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

def check_cuda():
    """Check if CUDA is available and get version"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        return cuda_available, cuda_version
    except ImportError:
        return False, None

def get_torch_installation():
    """Get appropriate PyTorch installation string"""
    cuda_available, cuda_version = check_cuda()
    
    if cuda_available and cuda_version:
        # Install CUDA-enabled PyTorch
        if cuda_version.startswith("11.8"):
            return "torch>=2.0.0"
        elif cuda_version.startswith("12.1"):
            return "torch>=2.0.0"
        else:
            return "torch>=2.0.0"
    else:
        # CPU-only
        return "torch>=2.0.0"

class CustomInstall(install):
    """Custom installation with PyTorch checks"""
    def run(self):
        print("\n🔍 Checking PyTorch/CUDA availability...")
        cuda_available, cuda_version = check_cuda()
        
        if cuda_available:
            print(f"✅ CUDA detected: {cuda_version}")
            print("   Installing with GPU support")
        else:
            print("⚠️  CUDA not detected")
            print("   Installing CPU-only version")
        
        super().run()

class CustomBuildExt(build_ext):
    """Custom build extension for C++ components"""
    def build_extensions(self):
        # Customize for PyTorch extensions if needed
        super().build_extensions()

# Read long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies
base_requirements = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.4.0",
    "numpy>=1.24.0",
    "pillow>=10.0.0",
    "python-multipart>=0.0.6",
    "websockets>=12.0",
    "aiofiles>=23.2.0",
    "python-dotenv>=1.0.0",
]

# Get appropriate PyTorch version
torch_requirement = get_torch_installation()
base_requirements.insert(0, torch_requirement)

# Optional dependencies
extra_requirements = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pytest-cov>=4.0.0",
        "pre-commit>=3.5.0",
    ],
    "gpu": [
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "nvidia-cuda-nvrtc-cu12>=12.1.105",
        "nvidia-cuda-runtime-cu12>=12.1.105",
    ],
    "full": [
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "ultralytics>=8.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "opencv-python>=4.8.0",
        "tensorboard>=2.13.0",
    ],
    "analytics": [
        "prometheus-client>=0.18.0",
        "grafana-sdk>=0.1.0",
        "plotly>=5.17.0",
    ]
}

# Check if we have C++ extensions
extensions = []
# Example: Uncomment if you have C++ extensions
# extensions.append(
#     Extension(
#         "babybionn._cuda_ops",
#         sources=["babybionn/cuda_ops.cpp"],
#         include_dirs=["./babybionn"],
#         extra_compile_args=["-O3", "-std=c++17"],
#     )
# )

setup(
    # Metadata is in pyproject.toml, but we keep some here for compatibility
    name="babybionn",
    version="2.0.0",
    author="BabyBIONN Team",
    author_email="contact@babybionn.ai",
    description="Enhanced BabyBIONN with Real Learning and Dynamic VNI Networking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/babybionn",
    
    # Find packages
    packages=find_packages(include=["babybionn", "babybionn.*"]),
    
    # Package data
    include_package_data=True,
    package_data={
        "babybionn": [
            "static/*",
            "static/**/*",
            "models/*.json",
            "config/*.yaml",
            "config/**/*.yaml",
        ],
    },
    
    # Dependencies
    install_requires=base_requirements,
    extras_require=extra_requirements,
    
    # C++ extensions if any
    ext_modules=extensions,
    
    # Commands
    cmdclass={
        "build_ext": CustomBuildExt,
        "install": CustomInstall,
    },
    
    # Additional metadata
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai, neural-networks, cognitive-architecture, multi-agent, vni",
    project_urls={
        "Documentation": "https://docs.babybionn.ai",
        "Source": "https://github.com/yourusername/babybionn",
        "Tracker": "https://github.com/yourusername/babybionn/issues",
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "babybionn=babybionn.main:main",
            "babybionn-api=babybionn.main:main",
            "babybionn-cli=babybionn.cli:main",
        ],
    },
) 
