# Face Cropper — AI Training Dataset

โปรแกรม crop หน้าคนออกจากรูปภาพ พร้อม AI Upscale สำหรับทำ dataset เทรน AI (LoRA / Dreambooth)

## Features

- ตรวจจับหน้าด้วย **YuNet DNN** (แม่นกว่า Haar cascade)
- **AI Upscale** ด้วย Real-ESRGAN x4
  - **GPU**: โหลด `.pth` ด้วย PyTorch CUDA — รองรับ RTX 50xx (Blackwell sm_120)
  - **CPU fallback**: โหลด `.onnx` ด้วย ONNX Runtime
- เลือก output size ได้ (512 / 768 / 1024 / 1280 / 1536 / 2048 px)
- คง **aspect ratio** — ด้านที่ยาวสุดเท่ากับค่าที่เลือก
- ปรับ **face padding** ได้ (1.0 = แน่น, 2.0 = หลวม)
- GUI พร้อม progress bar และ log แบบ real-time

## Files

| ไฟล์ | คำอธิบาย |
|------|-----------|
| `face_cropper_gui.py` | GUI หลัก (tkinter) |
| `crop_faces.py` | CLI script สำหรับรันตรงๆ |
| `convert_pth_to_onnx.py` | แปลง Real-ESRGAN `.pth` → `.onnx` |
| `FaceCropper.spec` | PyInstaller spec สำหรับ build `.exe` (รวม PyTorch GPU) |
| `build_exe.bat` | Build script |

## Setup

```bash
pip install opencv-python pillow onnxruntime pyinstaller torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### โมเดลที่ต้องดาวน์โหลดแยก

| ไฟล์ | แหล่ง |
|------|-------|
| `face_detection_yunet_2023mar.onnx` | [opencv_zoo](https://github.com/opencv/opencv_zoo) |
| `haarcascade_frontalface_default.xml` | มากับ `opencv-python` (`cv2.data.haarcascades`) |
| `RealESRGAN_x4plus.pth` | [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) |

## Build .exe (GPU)

spec ปัจจุบันรวม PyTorch + CUDA DLLs ไว้ใน exe เลย ไม่ต้องติดตั้ง Python บนเครื่องปลายทาง

```bash
build_exe.bat
```

ได้ `dist/FaceCropper.exe` — ขนาด ~3 GB รวม GPU runtime ครบ

## การใช้งาน

1. เลือก **Input Folder** (folder รูปต้นฉบับ)
2. เลือก **Output Folder**
3. ตั้ง **Max size** เช่น 1024 px
4. ตั้ง **Face padding** แนะนำ 1.6
5. เปิด **AI Upscale** — model `.pth` จะใช้ GPU อัตโนมัติ (ขึ้น `[CUDA]` ใน log)
6. กด **Start**

## GPU Support

| GPU | onnxruntime-gpu | PyTorch |
|-----|----------------|---------|
| RTX 30xx / 40xx | ✓ | ✓ |
| RTX 50xx (Blackwell sm_120) | ✗ | ✓ (2.12+cu128) |

→ ใช้ `.pth` + TorchUpscaler เพื่อรองรับ RTX 50xx
