"""
BiyoVes - Biyometrik Fotoğraf İşleme Kütüphanesi
Pure Python versiyonu - Tüm platformlarda (Windows, macOS, Linux) çalışır
"""

from setuptools import setup, find_packages
from pathlib import Path

# README dosyasını oku
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="biyoves",
    version="0.5.0",
    author="BiyoVes",
    description="Biyometrik fotoğraf işleme kütüphanesi - Kolay kullanımlı vesikalık ve biyometrik fotoğraf hazırlama aracı",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aytugyuruk/biyoves",
    packages=find_packages(),
    package_data={
        "biyoves": ["modnet.onnx"],
    },
    include_package_data=True,
    install_requires=[
        "opencv-python-headless>=4.5.0",  # GUI olmadan, 20-30MB daha hafif
        "numpy>=1.19.0",
        "mediapipe>=0.10.0,<0.11.0",  # Spesifik versiyonla minimal indirim
        "onnxruntime>=1.10.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="biyometrik fotoğraf vesikalık image processing face detection",
)

