@echo off
REM Run the all-in-one SAFE-T verification script.
REM Examples:
REM   verify_all.bat
REM   verify_all.bat --bundle dist\SAFE-T
REM   verify_all.bat --skip-mfa-align

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo ERROR: .venv\Scripts\python.exe not found. Please set up the virtual environment first.
    pause
    exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0verify_all.py" %*
if errorlevel 1 (
    echo.
    echo Verification failed. See summary above.
    pause
    exit /b 1
)
echo.
echo Verification succeeded!
pause
