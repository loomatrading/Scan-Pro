@echo off
setlocal
python -m pip install --upgrade pip
python -m pip install -r requirements-source.txt
if errorlevel 1 (
  echo.
  echo Installation failed.
  pause
  exit /b 1
)
echo.
echo Source dependencies installed successfully.
pause
