import os
import re
import subprocess
import sys

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
]

def prompt_for_gpu():
    """Ask the user if they have a GPU."""
    while True:
        response = input("Does your system have a GPU? (y/n): ").strip().lower()
        if response in ('y', 'n'):
            install_requires.append("onnxruntime-gpu" if response == 'y' else "onnxruntime")
            print(f"install_requires :{install_requires}")
            return response == 'y'
        print("Please enter 'y' or 'n'.")

def prompt_for_cuda_version():
    """Ask the user for their CUDA version and validate the input format."""
    while True:
        cuda_version = input("Please enter your CUDA version (e.g., 11.8 should be entered as 118): ").strip()
        if re.match(r'^\d{2,3}$', cuda_version):
            return cuda_version
        print("Invalid format, please enter a number like '124' or '118'.")

def main():
    # Determine if the user has a GPU
    has_gpu = prompt_for_gpu()

    if has_gpu:
        # If the user has a GPU, ask for the CUDA version
        cuda_version = prompt_for_cuda_version()
        # Write the CUDA version to a file
        with open('cuda_version.txt', 'w') as f:
            f.write(cuda_version)

    # Execute pip install -e .
    try:
        print("Installing the package, please wait...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("Installation completed.")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

