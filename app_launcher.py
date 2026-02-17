#!/usr/bin/env python3
"""
AI Product and Price Scrapper - Standalone Launcher
This script serves as the entry point for the standalone executable.
"""

import os
import sys
import time
import signal
import psutil
import subprocess
import threading
import pkg_resources
import importlib.util
from pathlib import Path
import streamlit.cli

# Define required packages
REQUIRED_PACKAGES = {
    "streamlit": "streamlit",
    "psutil": "psutil",
    "PyPDF2": "PyPDF2",
    "fitz": "pymupdf",
    "pdfplumber": "pdfplumber",
    "loguru": "loguru",
    "beautifulsoup4": "beautifulsoup4",
    "requests": "requests"
}

def is_package_installed(package_name):
    """Check if a Python package is installed."""
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False
    
def install_package(package_name):
    """Install a Python package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def ensure_dependencies():
    """Ensure all required dependencies are installed."""
    missing_packages = []
    
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            # Try to import the module
            if importlib.util.find_spec(module_name) is None:
                missing_packages.append(package_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"Installing missing dependencies: {', '.join(missing_packages)}")
        for package in missing_packages:
            print(f"Installing {package}...")
            install_package(package)
        
        # Verify installation
        still_missing = []
        for module_name, package_name in REQUIRED_PACKAGES.items():
            if importlib.util.find_spec(module_name) is None:
                still_missing.append(package_name)
        
        if still_missing:
            print(f"Error: Failed to install some dependencies: {', '.join(still_missing)}")
            return False
    
    return True

def kill_processes_on_ports(ports):
    """Kill any processes running on specified ports"""
    killed = []
    
    for port in ports:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any(f'--server.port={port}' in cmd for cmd in cmdline if cmd):
                    pid = proc.info['pid']
                    name = proc.info['name']
                    try:
                        os.kill(pid, signal.SIGTERM)
                        killed.append(f"Terminated {name} (PID: {pid}) on port {port}")
                    except Exception as e:
                        print(f"Failed to kill process {pid}: {str(e)}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    
    return killed

def launch_app(app_file, port=None):
    """Launch a Streamlit application"""
    try:
        # Get the full path to the app file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(current_dir, app_file)
        
        # Verify the file exists
        if not os.path.exists(app_path):
            print(f"Application file not found: {app_path}")
            return False
        
        # Build the arguments for streamlit.cli.main
        args = ["streamlit", "run", app_path]
        if port:
            args.extend(["--server.port", str(port)])
            
            # First, clean up any existing processes on this port
            kill_processes_on_ports([port])
            
        # Call streamlit CLI directly
        sys.argv = args
        streamlit.cli.main()
            
        return True
    
    except Exception as e:
        print(f"Error launching application: {str(e)}")
        return False

def main():
    """Main entry point"""
    # Show welcome message
    print("="*50)
    print("    AI Product and Price Scrapper - Launcher")
    print("="*50)
    print("\nPreparing to launch...\n")
    
    # Check dependencies
    print("Checking dependencies...")
    if not ensure_dependencies():
        print("Error: Failed to install required dependencies.")
        input("Press Enter to exit...")
        return
    
    print("All dependencies installed successfully.")
    
    # Launch the PDF Processor directly
    print("\nLaunching PDF Processor...\n")
    launch_app("pdf_app.py", 8602)

if __name__ == "__main__":
    main() 