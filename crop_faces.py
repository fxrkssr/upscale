import cv2
import os
from PIL import Image
from pathlib import Path

INPUT_DIR = Path(__file__).parent
OUTPUT_DIR = INPUT_DIR / "faces_1024"
MODEL_PATH = str(INPUT_DIR / "face_detection_yunet_2023mar.onnx")
TARGET_SIZE = 1024
# padding factor: ขยาย bbox เพื่อให้เห็นผม + คอ + บ่าเล็กน้อย
PADDING_FACTOR = 1.6

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def crop_face(image_path, detector):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, "cannot read"

    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)

    if faces is None or len(faces) == 0:
        # fallback: Haar cascade ถ้า YuNet ตรวจไม่เจอ
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(found) == 0:
            return None, "no face"
        # เลือก face ที่ใหญ่สุด
        x, y, fw, fh = max(found, key=lambda r: r[2] * r[3])
        bx, by, bw, bh = float(x), float(y), float(fw), float(fh)
    else:
        # เลือก face ที่ confidence สูงสุด (column 14)
        best = faces[faces[:, 14].argmax()]
        bx, by, bw, bh = best[0], best[1], best[2], best[3]

    cx = bx + bw / 2
    cy = by + bh / 2

    # ขยาย bbox เป็นสี่เหลี่ยมจัตุรัสด้วย padding
    size = max(bw, bh) * PADDING_FACTOR
    x1 = max(0, int(cx - size / 2))
    y1 = max(0, int(cy - size / 2))
    x2 = min(w, int(cx + size / 2))
    y2 = min(h, int(cy + size / 2))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, "empty crop"

    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    pil = pil.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    return pil, "ok"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6, nms_threshold=0.3)

    images = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in EXTS]
    images.sort()
    print(f"พบรูปทั้งหมด: {len(images)} ไฟล์")

    ok, skipped = 0, []

    for i, img_path in enumerate(images, 1):
        out_path = OUTPUT_DIR / (img_path.stem + ".jpg")
        cropped, status = crop_face(img_path, detector)

        if cropped is None:
            print(f"[{i:3}/{len(images)}] ข้าม  {img_path.name} ({status})")
            skipped.append(img_path.name)
        else:
            cropped.save(str(out_path), "JPEG", quality=95, subsampling=0)
            print(f"[{i:3}/{len(images)}] OK    {img_path.name}")
            ok += 1

    print(f"\nสรุป: สำเร็จ {ok} / {len(images)} รูป")
    if skipped:
        print(f"ข้ามไป {len(skipped)} รูป:")
        for name in skipped:
            print(f"  - {name}")
    print(f"\nผลลัพธ์อยู่ที่: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
