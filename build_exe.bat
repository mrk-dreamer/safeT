@echo off
REM Build the zero-setup SAFE-T standalone bundle (and zip archive).
REM Pass -Installer as well if Inno Setup 6 is installed.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\build_bundle.ps1" -Zip %*
if errorlevel 1 (
    echo.
    echo Build failed. See error message above.
    pause
    exit /b 1
)
echo.
echo Build succeeded! Output is in dist\SAFE-T (and dist\SAFE-T.zip)
pause
