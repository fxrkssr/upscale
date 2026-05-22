# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

ORT_DIR = r"C:\Users\KssR\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\onnxruntime"
CV2_DIR = r"C:\Users\KssR\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\cv2"

# รวม torch ทั้งหมด (Python files + CUDA DLLs)
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')

binaries = [
    (os.path.join(ORT_DIR, "capi", "onnxruntime.dll"),                  "onnxruntime/capi"),
    (os.path.join(ORT_DIR, "capi", "onnxruntime_providers_shared.dll"), "onnxruntime/capi"),
    (os.path.join(ORT_DIR, "capi", "onnxruntime_providers_cuda.dll"),   "onnxruntime/capi"),
    (os.path.join(CV2_DIR, "cv2.pyd"),                                   "cv2"),
    (os.path.join(CV2_DIR, "opencv_videoio_ffmpeg4130_64.dll"),          "cv2"),
] + torch_binaries

datas = [
    ("face_detection_yunet_2023mar.onnx",   "."),
    ("haarcascade_frontalface_default.xml", "."),
    (r"D:\Downloads\RealESRGAN_x4plus.pth", "."),   # GPU model (TorchUpscaler)
    ("realesrgan_x4.onnx",                  "."),   # CPU fallback (OnnxUpscaler)
    (os.path.join(CV2_DIR, "data"),          "cv2/data"),
    (ORT_DIR,                                "onnxruntime"),
] + torch_datas

a = Analysis(
    ["face_cropper_gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "PIL._tkinter_finder",
        "onnxruntime",
        "onnxruntime.capi",
        "onnxruntime.capi.onnxruntime_inference_collection",
        "cv2",
        "torch",
        "torch.nn",
        "torch.nn.functional",
    ] + torch_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tensorflow", "matplotlib", "scipy", "pandas"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FaceCropper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
