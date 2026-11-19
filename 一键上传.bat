@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal

REM ASCII-only file: Chinese text is printed by scripts\push.py

set "PY="
if exist "C:\ProgramData\anaconda3\python.exe" set "PY=C:\ProgramData\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY (
  where py >nul 2>nul
  if not errorlevel 1 set "PY=py"
)
if not defined PY (
  echo  [ERROR] Python not found.
  pause
  exit /b 1
)

"%PY%" scripts\push.py --force %*
set "RC=%ERRORLEVEL%"
echo.
echo  Press any key to close...
pause >nul
exit /b %RC%
