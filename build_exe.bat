@echo off
setlocal
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if not exist models mkdir models
if not exist models\EDSR_x2.pb (
  echo Downloading EDSR AI model...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb' -OutFile 'models/EDSR_x2.pb'"
)
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
python -m PyInstaller --noconfirm --clean --onefile --windowed --name ScanPro --add-data "models;models" app.py
if errorlevel 1 exit /b 1
echo.
echo Build complete: dist\ScanPro.exe
pause
