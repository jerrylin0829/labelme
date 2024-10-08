import os
import subprocess
import sys
import re
import argparse
import platform
from packaging import version
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_nvcc_version(nvcc_path):
    try:
        result = subprocess.run([nvcc_path, '--version'], capture_output=True, text=True, check=True)
        version_match = re.search(r'release (\d+\.\d+)', result.stdout)
        if version_match:
            return version_match.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to run nvcc at {nvcc_path}: {e}")
    return None

def detect_cuda_version(cuda_dir=None):
    cuda_versions = []

    if platform.system() == "Linux":
        cuda_base_path = cuda_dir or os.environ.get('CUDA_HOME') or '/usr/local'
        for entry in os.listdir(cuda_base_path):
            if entry.startswith("cuda"):
                nvcc_path = os.path.join(cuda_base_path, entry, "bin", "nvcc")
                if os.path.exists(nvcc_path):
                    ver = get_nvcc_version(nvcc_path)
                    if ver:
                        cuda_versions.append((ver, entry))
    elif platform.system() == "Windows":
        cuda_base_path = cuda_dir or os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        nvcc_path = os.path.join(cuda_base_path, "bin", "nvcc.exe")
        if os.path.exists(nvcc_path):
            ver = get_nvcc_version(nvcc_path)
            if ver:
                cuda_versions.append((ver, "CUDA"))

    if not cuda_versions:
        logger.info("No CUDA installations detected.")
        return None

    latest_version_info = max(cuda_versions, key=lambda x: version.parse(x[0]))
    latest_version, latest_entry = latest_version_info
    logger.info(f"Using CUDA version {latest_version} from {latest_entry}")
    return latest_version

def validate_cuda_version(cuda_ver, cuda_dir=None):
    detected_version = detect_cuda_version(cuda_dir)

    if cuda_ver and detected_version:
        if cuda_ver != detected_version:
            logger.warning(f"Specified CUDA version {cuda_ver} is not available, using detected version {detected_version}.")
            return detected_version
    elif cuda_ver and not detected_version:
        logger.warning(f"Specified CUDA version {cuda_ver} is not available. No CUDA installation detected, proceeding with CPU version.")
        return None

    return detected_version

def install_package(cuda_version):
    try:
        if cuda_version:
            logger.info(f"Installing package with CUDA {cuda_version} support...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "package_name[cuda]"])
        else:
            logger.info("Installing package without CUDA support...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "package_name"])
        logger.info("Installation completed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed: {e}")
        sys.exit(1)

def main(cuda_ver=None, cuda_dir=None):
    cuda_version = validate_cuda_version(cuda_ver, cuda_dir)
    logger.info(f"Final CUDA version used for installation: {cuda_version}")
    install_package(cuda_version)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install package with optional CUDA support.")
    parser.add_argument("--cuda_ver", type=str, help="Specify the CUDA version (e.g., 11.8)")
    parser.add_argument("--cuda_dir", type=str, help="Specify the CUDA installation folder")
    args = parser.parse_args()

    main(args.cuda_ver, args.cuda_dir)
