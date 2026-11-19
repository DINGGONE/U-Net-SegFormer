@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal

REM  This file is intentionally ASCII-only.
REM  Windows cmd.exe parses non-ASCII bytes in .bat files unreliably,
REM  so all Chinese text is printed by scripts\first_run.py instead.

set "PY="
if exist "C:\ProgramData\anaconda3\python.exe" set "PY=C:\ProgramData\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python39\python.exe"

if not defined PY (
  where py >nul 2>nul
  if not errorlevel 1 set "PY=py"
)

if not defined PY (
  echo.
  echo  [ERROR] Python not found.
  echo  Please install Anaconda: https://www.anaconda.com/download
  echo  then double-click this file again.
  echo.
  if "%~1"=="" pause
  exit /b 1
)

"%PY%" scripts\first_run.py %*
set "RC=%ERRORLEVEL%"

if "%~1"=="" (
  echo.
  echo  Press any key to close this window...
  pause >nul
)
exit /b %RC%
