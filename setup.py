from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aruco-barcode-detector",
    version="0.1.0",
    author="Your Name",
    description="Object detection model training for ArUco barcodes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "opencv-contrib-python>=4.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
)
