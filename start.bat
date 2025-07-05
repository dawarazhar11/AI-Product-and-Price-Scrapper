@echo off
echo Starting TonyScrapper Tools...
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
    pip install streamlit psutil PyPDF2 pymupdf pdfplumber loguru
)

:: Run the simple launcher
echo Launching TonyScrapper Tools...
streamlit run simple_launcher.py --server.port=8600

:: Keep the window open if there's an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred. Please check the output above.
    echo Press any key to exit...
    pause >nul
) 