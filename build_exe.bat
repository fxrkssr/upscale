@echo off
echo Building FaceCropper.exe ...
cd /d "%~dp0"

pyinstaller ^
  --onefile ^
  --windowed ^
  --name "FaceCropper" ^
  --add-data "face_detection_yunet_2023mar.onnx;." ^
  --add-data "haarcascade_frontalface_default.xml;." ^
  --hidden-import "PIL._tkinter_finder" ^
  --collect-all cv2 ^
  face_cropper_gui.py

echo.
if exist dist\FaceCropper.exe (
    echo SUCCESS: dist\FaceCropper.exe
) else (
    echo FAILED: ดู error ด้านบน
)
pause
