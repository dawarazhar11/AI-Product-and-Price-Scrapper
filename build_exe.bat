@echo off
echo Building TonyScrapper PDF Processor Executable...
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

:: Check if PyInstaller is installed
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

:: Create a directory for the build
if not exist builds (
    mkdir builds
)

:: Build the executable
echo Building executable...
pyinstaller --onefile --name "TonyScrapper_PDF_Processor" ^
    --add-data "pdf_app.py;." ^
    --add-data "pdf_handler.py;." ^
    --add-data "requirements.txt;." ^
    --add-data "README.md;." ^
    --add-data ".gitignore;." ^
    --icon=NONE ^
    --windowed ^
    app_launcher.py

:: Copy necessary files to the dist folder
echo Copying necessary files to the dist folder...
copy pdf_app.py dist\ >nul
copy pdf_handler.py dist\ >nul
if exist requirements.txt copy requirements.txt dist\ >nul
if exist README.md copy README.md dist\ >nul

:: Create a simple launcher batch file in the dist folder
echo @echo off > dist\TonyScrapper_PDF_Processor.bat
echo cd /d "%%~dp0" >> dist\TonyScrapper_PDF_Processor.bat
echo start "" "TonyScrapper_PDF_Processor.exe" >> dist\TonyScrapper_PDF_Processor.bat

echo.
echo Build complete! The executable is located in the dist folder.
echo You can run the application by double-clicking TonyScrapper_PDF_Processor.exe
echo.
echo Press any key to exit...
pause >nul 