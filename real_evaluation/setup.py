from setuptools import setup
import os
import subprocess
import sys

def install_requirements(requirements_file):
    if requirements_file.endswith('.txt'):
        subprocess.check_call([os.sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
    else:
        print("incoorect file extension")
# Specify which file you want to install dependencies from (either 'requirements.txt' or 'environment.yml')
requirements_file = 'requirements.txt'  # Change this to 'environment.yml' if you want to use the YAML file
# Define the pip command
command = [
    sys.executable,  # This ensures the command is run with the current Python interpreter
    "-m", "pip",     # Use pip as a module
    "install",
    "torch==2.2.1",
    "torchvision==0.17.1",
    "torchaudio==2.2.1",
    "--index-url", "https://download.pytorch.org/whl/cu121"
]

# Execute the command
subprocess.check_call(command)
# First, install the dependencies from the selected file
install_requirements(requirements_file)

