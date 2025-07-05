@echo off
echo Starting TonyScrapper PDF Processor...
echo.

:: Check if Python is available in the path
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in the PATH.
    echo Please install Python or add it to your PATH.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

:: Check if the virtual environment exists
if exist venv\Scripts\activate.bat (
    echo Using existing virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Creating new virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    
    echo Installing required packages...
    pip install streamlit psutil PyPDF2 pymupdf pdfplumber loguru beautifulsoup4 requests
)

:: Run the PDF processor directly
echo Launching PDF Processor...
python app_launcher.py

:: Keep the window open if there's an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred. Please check the output above.
    echo Press any key to exit...
    pause >nul
) 