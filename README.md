# Face Cropper — AI Training Dataset

โปรแกรม crop หน้าคนออกจากรูปภาพ พร้อม AI Upscale สำหรับทำ dataset เทรน AI (LoRA / Dreambooth)

## Features

- ตรวจจับหน้าด้วย **YuNet DNN** (แม่นกว่า Haar cascade)
- **AI Upscale** ด้วย Real-ESRGAN x4 ผ่าน ONNX Runtime
- รองรับ **GPU (CUDA)** อัตโนมัติ ถ้าไม่มีจะ fallback CPU
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
| `FaceCropper.spec` | PyInstaller spec สำหรับ build `.exe` |
| `build_exe.bat` | Build script |

## Setup

```bash
pip install opencv-python mediapipe pillow onnxruntime-gpu pyinstaller
```

### โมเดลที่ต้องดาวน์โหลดแยก

| ไฟล์ | แหล่ง |
|------|-------|
| `face_detection_yunet_2023mar.onnx` | [opencv_zoo](https://github.com/opencv/opencv_zoo) |
| `haarcascade_frontalface_default.xml` | มากับ `opencv-python` (`cv2.data.haarcascades`) |
| `realesrgan_x4.onnx` | แปลงจาก `.pth` ด้วย `convert_pth_to_onnx.py` |

### แปลง Real-ESRGAN model

```bash
python convert_pth_to_onnx.py RealESRGAN_x4plus.pth realesrgan_x4.onnx
```

## Build .exe

```bash
build_exe.bat
```

ได้ `dist/FaceCropper.exe` — standalone ไม่ต้องติดตั้ง Python

## การใช้งาน

1. เลือก **Input Folder** (folder รูปต้นฉบับ)
2. เลือก **Output Folder**
3. ตั้ง **Max size** เช่น 1024 px
4. ตั้ง **Face padding** แนะนำ 1.6
5. เปิด **AI Upscale** และตั้ง threshold = `0` เพื่อ upscale ทุกรูป
6. กด **Start**
