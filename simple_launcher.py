#!/usr/bin/env python3
"""
AI Product and Price Scrapper - Simple Minimalist Launcher
"""

import os
import sys
import signal
import psutil
import subprocess
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="AI Product and Price Scrapper",
    page_icon="🛠️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Basic CSS
st.markdown("""
<style>
.header {
    text-align: center;
    padding: 1rem 0;
}
.btn-container {
    margin: 2rem 0;
    text-align: center;
}
.description {
    margin: 1rem 0;
    text-align: center;
    color: #666;
}
.divider {
    margin: 2rem 0;
    border-top: 1px solid #ddd;
}
.footer {
    margin-top: 2rem;
    text-align: center;
    font-size: 0.8rem;
    color: #888;
}
</style>
""", unsafe_allow_html=True)

# Function to kill processes using specific ports
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
                        st.error(f"Failed to kill process {pid}: {str(e)}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    
    return killed

# Launch application function
def launch_app(app_file, port):
    """Launch a Streamlit application"""
    try:
        # First, clean up any existing processes on this port
        kill_processes_on_ports([port])
        
        # Get the full path to the app file
        current_dir = os.getcwd()
        app_path = os.path.join(current_dir, app_file)
        
        # Verify the file exists
        if not os.path.exists(app_path):
            st.error(f"Application file not found: {app_path}")
            return False
        
        # Build the command
        cmd = [
            sys.executable,
            "-m", 
            "streamlit", 
            "run", 
            app_path,
            f"--server.port={port}"
        ]
        
        # Create proper startup info for Windows
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
            subprocess.Popen(cmd, startupinfo=startupinfo)
        else:
            subprocess.Popen(cmd)
            
        return True
    
    except Exception as e:
        st.error(f"Error launching application: {str(e)}")
        return False

# Header
st.markdown("<h1 class='header'>AI Product and Price Scrapper</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Select a tool to launch:</p>", unsafe_allow_html=True)

# Web Scraper Section
st.markdown("## Web Scraper")
st.markdown("Extract data from websites and process it")
col1a, col1b = st.columns([3, 1])
with col1a:
    if st.button("Launch Web Scraper", key="web_scraper", use_container_width=True):
        with st.spinner("Starting Web Scraper..."):
            success = launch_app("app.py", 8601)
            if success:
                st.success("Web Scraper started!")
                st.markdown(f"[Open Web Scraper](http://localhost:8601)")

# Divider
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# PDF Processor Section
st.markdown("## PDF Processor")
st.markdown("Extract and process data from PDF files")
col2a, col2b = st.columns([3, 1])
with col2a:
    if st.button("Launch PDF Processor", key="pdf_processor", use_container_width=True):
        with st.spinner("Starting PDF Processor..."):
            success = launch_app("pdf_app.py", 8602)
            if success:
                st.success("PDF Processor started!")
                st.markdown(f"[Open PDF Processor](http://localhost:8602)")

# Divider
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Cleanup Section
st.markdown("## Maintenance")
st.markdown("Manage running applications")
if st.button("Stop All Running Applications", key="stop_all"):
    with st.spinner("Stopping all applications..."):
        killed = kill_processes_on_ports([8501, 8502, 8601, 8602])
        if killed:
            for msg in killed:
                st.info(msg)
            st.success("All applications stopped")
        else:
            st.info("No running applications found")

# Footer
st.markdown("<div class='footer'>AI Product and Price Scrapper</div>", unsafe_allow_html=True) 