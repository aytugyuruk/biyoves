"""
Cython ile derlenmiş versiyon için setup.py - Universal (Tüm Platformlar)
Kodları gizlemek için kullanılır. Kullanıcılar kurulum sırasında derleyecek.
"""

from setuptools import setup, find_packages, Extension
from pathlib import Path

# Cython ve numpy'yi conditional import yap
try:
    from Cython.Build import cythonize
    import numpy
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    # Cython yoksa basit extension oluştur (kurulum sırasında Cython kurulacak)
    def cythonize(extensions, **kwargs):
        return extensions

# README dosyasını oku
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Cython ile derlenecek modüller
def get_extensions():
    if HAS_CYTHON:
        include_dirs = [numpy.get_include()]
    else:
        include_dirs = []
    
    return [
        Extension("biyoves.remove_bg", ["biyoves/remove_bg.py"], include_dirs=include_dirs),
        Extension("biyoves.corrector", ["biyoves/corrector.py"], include_dirs=include_dirs),
        Extension("biyoves.processor", ["biyoves/processor.py"], include_dirs=include_dirs),
        Extension("biyoves.layout", ["biyoves/layout.py"], include_dirs=include_dirs),
    ]

extensions = get_extensions()

setup(
    name="biyoves",
    version="0.4.0",
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
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }
    ) if HAS_CYTHON else extensions,
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "mediapipe>=0.10.0,<0.11.0",  # Spesifik versiyonla minimal indirim
        "onnxruntime>=1.10.0",
    ],
    setup_requires=[
        "Cython>=0.29.0",
        "numpy>=1.19.0",
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
    ],
    keywords="biyometrik fotoğraf vesikalık image processing face detection",
)

