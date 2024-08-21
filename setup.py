import distutils.spawn
import os
import re
import shlex
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup

# Run CUDA detection once during import
from install_labelme import detect_cuda_version

cuda_version = detect_cuda_version()  # Set this once to avoid recursion

def get_version():
    filename = "labelme/__init__.py"
    with open(filename) as f:
        match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M)
    if not match:
        raise RuntimeError("{} doesn't contain __version__".format(filename))
    version = match.groups()[0]
    return version

def get_install_requires():
    install_requires = [
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
        "opencv-python-headless"
    ]

    QT_BINDING = None

    for binding in ["PyQt5", "PySide2"]:
        try:
            __import__(binding)
            QT_BINDING = binding.lower()
            break
        except ImportError:
            continue

    if QT_BINDING is None:
        install_requires.append("PyQt5!=5.15.3,!=5.15.4")

    if os.name == "nt": # windows
        install_requires.append("colorama")

    print(f"CUDA version detected in setup: {cuda_version}")
    
    if cuda_version is not None:
        if cuda_version.startswith("12"):
            install_requires.append("onnxruntime-gpu>=1.18.0")
        elif cuda_version.startswith("11.8"):
            install_requires.append("onnxruntime-gpu>=1.17.0")
        elif cuda_version.startswith("11.6"):
            install_requires.append("onnxruntime-gpu>=1.14.0")
        elif cuda_version.startswith("11.4"):
            install_requires.append("onnxruntime-gpu>=1.10.0")
        else:
            install_requires.append("onnxruntime") 
    else:
        install_requires.append("onnxruntime")

    return install_requires

def install_package_from_git():
    # URL of the git repository
    git_repo_url = "https://github.com/facebookresearch/segment-anything.git"
    
    # Construct the pip install command
    command = f"{sys.executable} -m pip install git+{git_repo_url}"
    
    try:
        # Execute the command
        subprocess.check_call(command, shell=True)
        print("Package installed successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to install package from Git repository.")
        print(e)
        
        
def install_torch_packages():
    print(f"Detected CUDA version: {cuda_version}")

    if cuda_version is not None:
        url = f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", url
            ])
            print("Successfully installed PyTorch with CUDA support.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install PyTorch with CUDA: {e}")
            print("PyTorch installation skipped.")
    else:
        subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio"
            ])        
        print("CUDA not detected or specified, skipping PyTorch installation.")

def get_long_description():
    with open("README.md",'r', encoding='utf8') as f:
        long_description = f.read()
    try:
        import github2pypi

        return github2pypi.replace_url(
            slug="wkentaro/labelme", content=long_description, branch="main"
        )
    except ImportError:
        return long_description

def main():
    version = get_version()

    if len(sys.argv) > 1 and sys.argv[1] == "release":
        try:
            import github2pypi  # NOQA
        except ImportError:
            print(
                "Please install github2pypi\n\n\tpip install github2pypi\n",
                file=sys.stderr,
            )
            sys.exit(1)

        if not distutils.spawn.find_executable("twine"):
            print(
                "Please install twine:\n\n\tpip install twine\n",
                file=sys.stderr,
            )
            sys.exit(1)

        commands = [
            "git push origin main",
            "git tag v{:s}".format(version),
            "git push origin --tags",
            "python setup.py sdist",
            "twine upload dist/labelme-{:s}.tar.gz".format(version),
        ]
        for cmd in commands:
            print("+ {:s}".format(cmd))
            subprocess.check_call(shlex.split(cmd))
        sys.exit(0)

    setup(
        name="labelme",
        version=version,
        packages=find_packages(),
        description="Image Polygonal Annotation with Python",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author="Kentaro Wada",
        author_email="www.kentaro.wada@gmail.com",
        url="https://github.com/wkentaro/labelme",
        install_requires=get_install_requires(),
        license="GPLv3",
        keywords="Image Annotation, Machine Learning",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3 :: Only",
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
    try:
        install_package_from_git()
    except RuntimeError as e:
        print(e)
        print("An error occurred while trying to install segment-anything.")
       

    try:
        install_torch_packages()
    except RuntimeError as e:
        print(e)
        print("An error occurred while trying to install PyTorch.")

if __name__ == "__main__":
    main()
