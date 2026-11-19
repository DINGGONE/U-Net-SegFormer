@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal

REM  ASCII-only on purpose: cmd.exe mis-parses non-ASCII bytes in .bat files.
REM  All Chinese text is printed by scripts\upload_github.py

set "PY="
if exist "C:\ProgramData\anaconda3\python.exe" set "PY=C:\ProgramData\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY (
  where py >nul 2>nul
  if not errorlevel 1 set "PY=py"
)

if not defined PY (
  echo.
  echo  [ERROR] Python not found. Install Anaconda first.
  echo.
  pause
  exit /b 1
)

"%PY%" scripts\upload_github.py %*
set "RC=%ERRORLEVEL%"
echo.
echo  Press any key to close...
pause >nul
exit /b %RC%
