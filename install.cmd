@echo off
rem cacography installer for Windows: double-click this file.
rem It runs install.ps1 next to it (PowerShell may not run .ps1 files on a double-click).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
