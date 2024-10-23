import re
import sys
from setuptools import find_packages, setup

def get_version():
    filename = "labelme/__init__.py"
    with open(filename) as f:
        match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M)
    if not match:
        raise RuntimeError(f"{filename} doesn't contain __version__")
    version = match.groups()[0]
    return version

def get_long_description():
    with open("README.md", 'r', encoding='utf8') as f:
        long_description = f.read()
    return long_description

setup(
    name="labelme",
    version=get_version(),
    packages=find_packages(),
    description="Image Polygonal Annotation with Python",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Kentaro Wada",
    author_email="www.kentaro.wada@gmail.com",
    url="https://github.com/wkentaro/labelme",
    install_requires=[
        "gdown",
        "imgviz>=1.7.5",
        "matplotlib",
        "natsort>=7.1.0",
        "numpy<2.0.0",
        "Pillow>=2.8",
        "PyYAML",
        "qtpy!=1.11.2",
        "scikit-image",
        "termcolor",
        "pynvml",
        "opencv-python-headless",
    ],
    extras_require={
        "cuda": [
            "onnxruntime-gpu>=1.18.0",
            "torch",
            "torchvision",
            "torchaudio",
        ],
        "cpu": [
            "onnxruntime",
        ],
    },
    license="GPLv3",
    keywords="Image Annotation, Machine Learning",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    package_data={"labelme": ["icons/*", "config/*.yaml", "translate/*"]},
    entry_points={
        "console_scripts": [
            "labelme=labelme.__main__:main",
            "labelme_draw_json=labelme.cli.draw_json:main",
            "labelme_draw_label_png=labelme.cli.draw_label_png:main",
            "labelme_json_to_dataset=labelme.cli.json_to_dataset:main",
            "labelme_export_json=labelme.cli.export_json:main",
            "labelme_on_docker=labelme.cli.on_docker:main",
        ],
    },
)
