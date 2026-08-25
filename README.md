# ScanPro

A simple Windows document scanner inspired by the supplied reference UI.

## Features

- Large **+** button in the center to import an image.
- Automatic document-corner detection and perspective correction.
- Automatic scanner-style enhancement after import.
- Real **EDSR x2 AI** super-resolution model bundled in `models/EDSR_x2.pb`.
- **Original** restores the original image view.
- **Magic Pro AI** reruns automatic document correction and AI enhancement.
- Rotate left/right buttons.
- Save as JPG/PNG to Desktop by default.
- No PDF, filters, color controls, or manual adjustment panels.

## Run from source

```bat
python -m pip install -r requirements.txt
python app.py
```

## Build Windows EXE locally

Run:

```bat
build_exe.bat
```

The executable will be in `dist\\ScanPro.exe`.

## GitHub Actions

The repository includes `.github/workflows/build-windows.yml`.

Push the project to GitHub, then open **Actions → Build Windows EXE → Run workflow**.
The workflow downloads the EDSR model, builds the Windows executable with PyInstaller, and uploads `ScanPro-Windows.zip` as a workflow artifact.

PyInstaller creates a self-contained application bundle, so the target PC does not need Python or the Python packages installed.
