
# Testing Environment Setup Guide

This guide outlines the steps for setting up a testing environment for projects contained in this repository. It covers the installation of Python 3.9 on WSL2 Ubuntu 22.04, the setup of a Python virtual environment, and the installation of Jupyter.

## Python 3.9 Installation on WSL2 Ubuntu 22.04

Follow these steps to install Python 3.9:

1. **Add Python PPA and Update Packages**

   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt-get update
   ```

2. **Install Python 3.9 and Necessary Packages**

   ```bash
   sudo apt-get install --reinstall python3.9 python3.9-venv -y
   sudo apt-get install python3.9-distutils -y
   sudo apt-get install python3-pip -y
   ```

## Setting Up the Python Virtual Environment

1. **Create a Virtual Environment**

   ```bash
   python3.9 -m venv .venv
   ```

2. **Activate the Virtual Environment**

   ```bash
   source .venv/bin/activate
   ```

3. **Manual Installation of pip**

   If needed, manually install `pip` within the virtual environment:

   ```bash
   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   python get-pip.py
   ```

4. **Verify Installation**

   Ensure the `python3.9-venv` package is installed:

   ```bash
   sudo apt-get install python3.9-venv
   ```

## Jupyter Installation

After setting up the virtual environment, you can install Jupyter:

```bash
pip install jupyter
```
