import os
import subprocess
import sys
import re
import argparse

cuda_version = None  # Ensure cuda_version is a module-level global variable

def detect_cuda_version(cuda_dir=None):
    cuda_base_path = "/usr/local" if cuda_dir is None else cuda_dir
    cuda_versions = []

    for entry in os.listdir(cuda_base_path):
        if entry.startswith("cuda"):
            nvcc_path = os.path.join(cuda_base_path, entry, "bin", "nvcc")
            if os.path.exists(nvcc_path):
                try:
                    result = subprocess.run([nvcc_path, '--version'], capture_output=True, text=True, check=True)
                    version_match = re.search(r'release (\d+)\.(\d+)', result.stdout)
                    if version_match:
                        major = version_match.group(1)
                        minor = version_match.group(2)
                        cuda_versions.append((f"{major}.{minor}", entry))
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"Failed to run nvcc in {entry}: {e}")

    if not cuda_versions:
        print("No CUDA installations detected.")
        return None

    latest_version, latest_entry = max(cuda_versions, key=lambda x: float(x[0]))
    print(f"Using CUDA version {latest_version} from {latest_entry}")
    return latest_version

def validate_cuda_version(cuda_ver,cuda_dir=None):
    detected_version = detect_cuda_version(cuda_dir)
    specified_version = cuda_ver if cuda_ver else None

    if specified_version and detected_version:
        if specified_version != detected_version:
            print(f"Specified CUDA version {specified_version} is not available, using detected version {detected_version}.")
            return detected_version
    elif specified_version and not detected_version:
        print(f"Specified CUDA version {specified_version} is not available. No CUDA installation detected, proceeding with CPU version.")
        return None

    return detected_version

def main(cuda_ver=None,cuda_dir=None):
    global cuda_version  

    if cuda_ver:
        cuda_version = validate_cuda_version(cuda_ver,cuda_dir)
    else:
        cuda_version = detect_cuda_version(cuda_dir)
    
    print(f"Final CUDA version used for installation: {cuda_version}")

    # Call setup directly to avoid recursive calls
    try:
        print("Running setup.py, please wait...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("Installation completed.")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install labelme with CUDA support.")
    parser.add_argument("--cuda_ver", type=str, help="Specify the CUDA version (e.g., 11.8)")
    parser.add_argument("--cuda_dir", type=str, help="Specify the cuda installation folder")
    args = parser.parse_args()

    main(args.cuda_ver,args.cuda_dir)
