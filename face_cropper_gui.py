import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


FACE_MODEL      = resource_path("face_detection_yunet_2023mar.onnx")
HAAR_PATH       = resource_path("haarcascade_frontalface_default.xml")
BUILTIN_UPSCALE = resource_path("realesrgan_x4.onnx")


def _add_cuda_dll_path():
    """หา CUDA DLLs จาก PyTorch แล้วเพิ่มเข้า search path — ทำงานได้ทั้ง .py และ .exe"""
    def _try(lib):
        if os.path.isfile(os.path.join(lib, "cublasLt64_12.dll")):
            os.add_dll_directory(lib)
            return True
        return False

    # วิธีที่ 1: import torch ตรงๆ (ใช้ได้เมื่อรันเป็น .py)
    try:
        import torch as _t
        if _try(os.path.join(os.path.dirname(_t.__file__), "lib")):
            return
    except ImportError:
        pass

    # วิธีที่ 2: ถาม system Python ว่า torch อยู่ที่ไหน (ใช้ได้เมื่อรันเป็น .exe)
    try:
        import subprocess
        r = subprocess.run(
            ["python", "-c",
             "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))"],
            capture_output=True, text=True, timeout=15,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
        if r.returncode == 0 and _try(r.stdout.strip()):
            return
    except Exception:
        pass

    # วิธีที่ 3: ค้นหา torch lib ใน AppData (fallback)
    import glob
    local = os.environ.get("LOCALAPPDATA", "")
    for pat in [
        os.path.join(local, "Python", "*", "Lib", "site-packages", "torch", "lib"),
        os.path.join(local, "Programs", "Python", "*", "Lib", "site-packages", "torch", "lib"),
        r"C:\Python3*\Lib\site-packages\torch\lib",
    ]:
        for p in glob.glob(pat):
            if _try(p):
                return


# ─── AI Upscaler (ONNX) ────────────────────────────────────────────────────────

class OnnxUpscaler:
    """โหลด Real-ESRGAN หรือ ONNX SR model ใดก็ได้ (NCHW float32 in/out, range 0-1)"""

    def __init__(self, model_path):
        _add_cuda_dll_path()
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(model_path, providers=providers)
        self.device = self._sess.get_providers()[0].replace("ExecutionProvider", "")
        self._inp  = self._sess.get_inputs()[0].name
        self._out  = self._sess.get_outputs()[0].name
        self._scale = self._detect_scale()

    def _detect_scale(self):
        probe = np.zeros((1, 3, 16, 16), dtype=np.float32)
        out   = self._sess.run([self._out], {self._inp: probe})[0]
        return out.shape[2] // 16  # e.g. 64//16 = 4

    @property
    def scale(self):
        return self._scale

    def _infer(self, img_rgb: np.ndarray) -> np.ndarray:
        """img_rgb: HxWx3 uint8  →  upscaled HxWx3 uint8"""
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]          # NCHW
        y = self._sess.run([self._out], {self._inp: x})[0]
        y = np.squeeze(y, 0).transpose(1, 2, 0)       # HWC
        return np.clip(y * 255, 0, 255).astype(np.uint8)

    def upscale(self, img_bgr: np.ndarray, tile: int = 512, pad: int = 16) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        if max(h, w) <= tile:
            out_rgb = self._infer(img_rgb)
        else:
            out_rgb = self._tiled(img_rgb, tile, pad)

        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    def _tiled(self, img_rgb: np.ndarray, tile: int, pad: int) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        s    = self._scale
        out  = np.zeros((h * s, w * s, 3), dtype=np.uint8)

        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y1 = max(0, y - pad);  x1 = max(0, x - pad)
                y2 = min(h, y + tile + pad);  x2 = min(w, x + tile + pad)

                tile_in   = img_rgb[y1:y2, x1:x2]
                tile_out  = self._infer(tile_in)

                # ตัดส่วน padding ออกจาก tile output
                oy1 = (y - y1) * s;  ox1 = (x - x1) * s
                oy2 = oy1 + min(tile, h - y) * s
                ox2 = ox1 + min(tile, w - x) * s

                dy1 = y * s;  dx1 = x * s
                dy2 = dy1 + (oy2 - oy1)
                dx2 = dx1 + (ox2 - ox1)

                out[dy1:dy2, dx1:dx2] = tile_out[oy1:oy2, ox1:ox2]

        return out


# ─── App ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Cropper — AI Training Dataset")
        self.geometry("720x680")
        self.minsize(640, 560)
        self._running  = False
        self._upscaler = None   # cache โมเดลตลอด session
        self._build()

    # ─── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        pad = {"padx": 12, "pady": 4}

        # Input folder
        fi = ttk.LabelFrame(self, text=" Input Folder ")
        fi.pack(fill="x", **pad)
        self.v_in = tk.StringVar()
        ttk.Entry(fi, textvariable=self.v_in, width=64).pack(side="left", padx=6, pady=6)
        ttk.Button(fi, text="Browse…", command=lambda: self._browse_dir(self.v_in)).pack(side="left")

        # Output folder
        fo = ttk.LabelFrame(self, text=" Output Folder ")
        fo.pack(fill="x", **pad)
        self.v_out = tk.StringVar()
        ttk.Entry(fo, textvariable=self.v_out, width=64).pack(side="left", padx=6, pady=6)
        ttk.Button(fo, text="Browse…", command=lambda: self._browse_dir(self.v_out, is_out=True)).pack(side="left")

        # ── Crop settings ───────────────────────────────────────────────────────
        fs = ttk.LabelFrame(self, text=" Crop Settings ")
        fs.pack(fill="x", **pad)

        r1 = ttk.Frame(fs)
        r1.pack(fill="x", padx=8, pady=6)
        ttk.Label(r1, text="Max size (ด้านที่ยาวสุด):").pack(side="left")
        self.v_size = tk.StringVar(value="1024")
        ttk.Combobox(r1, textvariable=self.v_size, width=7,
                     values=["512","768","1024","1280","1536","2048"]).pack(side="left", padx=5)
        ttk.Label(r1, text="px").pack(side="left")
        ttk.Label(r1, text="      Face padding:").pack(side="left")
        self.v_pad = tk.StringVar(value="1.6")
        ttk.Entry(r1, textvariable=self.v_pad, width=5).pack(side="left", padx=4)
        ttk.Label(r1, text="(1.0=แน่น / 2.0=หลวม)", foreground="gray").pack(side="left")

        r2 = ttk.Frame(fs)
        r2.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(r2, text="Format:").pack(side="left")
        self.v_fmt = tk.StringVar(value="JPEG")
        ttk.Radiobutton(r2, text="JPEG", variable=self.v_fmt, value="JPEG").pack(side="left", padx=5)
        ttk.Radiobutton(r2, text="PNG",  variable=self.v_fmt, value="PNG").pack(side="left")
        ttk.Label(r2, text="     JPEG quality:").pack(side="left")
        self.v_q = tk.StringVar(value="95")
        ttk.Entry(r2, textvariable=self.v_q, width=4).pack(side="left", padx=4)

        # ── AI Upscale ─────────────────────────────────────────────────────────
        fu = ttk.LabelFrame(self, text=" AI Upscale (Real-ESRGAN / ONNX) ")
        fu.pack(fill="x", **pad)

        u1 = ttk.Frame(fu)
        u1.pack(fill="x", padx=8, pady=4)
        self.v_up_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(u1, text="เปิด AI Upscale", variable=self.v_up_on,
                        command=self._toggle_upscale).pack(side="left")
        ttk.Label(u1, text="   เฉพาะ crop ที่ด้านสั้น <").pack(side="left")
        self.v_up_thresh = tk.StringVar(value="512")
        self.e_thresh = ttk.Entry(u1, textvariable=self.v_up_thresh, width=6)
        self.e_thresh.pack(side="left", padx=3)
        ttk.Label(u1, text="px (0 = ทุกรูป)").pack(side="left")

        u2 = ttk.Frame(fu)
        u2.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(u2, text="ONNX model:").pack(side="left")
        # ถ้ามีโมเดลฝังมากับ exe ให้แสดงชื่อเลย
        default_model = BUILTIN_UPSCALE if os.path.exists(BUILTIN_UPSCALE) else ""
        self.v_up_model = tk.StringVar(value=default_model)
        self.e_upmodel = ttk.Entry(u2, textvariable=self.v_up_model, width=52, state="disabled")
        self.e_upmodel.pack(side="left", padx=5)
        self.btn_up_browse = ttk.Button(u2, text="Browse…",
                                        command=self._browse_model, state="disabled")
        self.btn_up_browse.pack(side="left")

        if os.path.exists(BUILTIN_UPSCALE):
            hint_txt = "  ✓ Real-ESRGAN x4 ฝังมากับโปรแกรมแล้ว (หรือ Browse เลือก model อื่น)"
            hint_color = "#7ecf7e"
        else:
            hint_txt = "  ดาวน์โหลดโมเดล: openmodeldb.info → Real-ESRGAN x4"
            hint_color = "#888"
        ttk.Label(fu, text=hint_txt, foreground=hint_color, font=("", 8)
                  ).pack(anchor="w", padx=8, pady=(0, 4))

        # ── Progress ───────────────────────────────────────────────────────────
        self.pb = ttk.Progressbar(self, mode="determinate")
        self.pb.pack(fill="x", padx=12, pady=(6, 0))
        self.v_status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.v_status, anchor="w").pack(fill="x", padx=14)

        # ── Log ────────────────────────────────────────────────────────────────
        lf = ttk.Frame(self)
        lf.pack(fill="both", expand=True, padx=12, pady=4)
        self.log = tk.Text(lf, height=9, state="disabled",
                           bg="#16213e", fg="#e0e0e0", font=("Consolas", 9),
                           relief="flat", bd=0)
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

        # ── Buttons ────────────────────────────────────────────────────────────
        fb = ttk.Frame(self)
        fb.pack(pady=8)
        self.btn_start = ttk.Button(fb, text="▶  Start", width=14, command=self._start)
        self.btn_start.pack(side="left", padx=6)
        self.btn_stop = ttk.Button(fb, text="■  Stop", width=10,
                                   command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        self.btn_open = ttk.Button(fb, text="Open Output Folder ▸",
                                   command=self._open_output, state="disabled")
        self.btn_open.pack(side="left", padx=6)

    # ─── helpers ───────────────────────────────────────────────────────────────

    def _toggle_upscale(self):
        st = "normal" if self.v_up_on.get() else "disabled"
        self.e_upmodel.configure(state=st)
        self.btn_up_browse.configure(state=st)
        self.e_thresh.configure(state=st)

    def _browse_dir(self, var, is_out=False):
        d = filedialog.askdirectory()
        if not d:
            return
        var.set(d)
        if not is_out and not self.v_out.get():
            self.v_out.set(os.path.join(d, f"faces_{self.v_size.get()}"))

    def _browse_model(self):
        f = filedialog.askopenfilename(
            title="เลือก ONNX model",
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")]
        )
        if f:
            self.v_up_model.set(f)
            self._upscaler = None  # reset cache เมื่อเปลี่ยนโมเดล

    def _log(self, msg, color=None):
        self.log.configure(state="normal")
        if color:
            tag = f"c_{color}"
            self.log.tag_configure(tag, foreground=color)
            self.log.insert("end", msg + "\n", tag)
        else:
            self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _open_output(self):
        d = self.v_out.get().strip()
        if d and os.path.isdir(d):
            os.startfile(d)

    # ─── control ───────────────────────────────────────────────────────────────

    def _start(self):
        inp = self.v_in.get().strip()
        out = self.v_out.get().strip()
        if not inp or not out:
            messagebox.showwarning("Missing", "กรุณาเลือก Input และ Output folder")
            return
        if not os.path.isdir(inp):
            messagebox.showerror("Error", f"ไม่พบ Input folder:\n{inp}")
            return
        try:
            max_size = int(self.v_size.get())
            padding  = float(self.v_pad.get())
            quality  = int(self.v_q.get())
            assert 1 <= quality <= 100
            assert 0.5 <= padding <= 5.0
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        # validate upscale settings
        use_up    = self.v_up_on.get()
        up_model  = self.v_up_model.get().strip()
        up_thresh = 0
        if use_up:
            if not up_model:
                messagebox.showwarning("Upscale", "กรุณาเลือก ONNX model ก่อน")
                return
            if not os.path.isfile(up_model):
                messagebox.showerror("Upscale", f"ไม่พบไฟล์ model:\n{up_model}")
                return
            try:
                up_thresh = int(self.v_up_thresh.get())
            except Exception:
                up_thresh = 512

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_open.configure(state="disabled")
        self.pb.configure(value=0)

        threading.Thread(
            target=self._run,
            args=(inp, out, max_size, padding, self.v_fmt.get(), quality,
                  use_up, up_model, up_thresh),
            daemon=True,
        ).start()

    def _stop(self):
        self._running = False

    # ─── worker ────────────────────────────────────────────────────────────────

    def _run(self, input_dir, output_dir, max_size, padding, fmt, quality,
             use_up, up_model_path, up_thresh):

        def log(msg, color=None):
            self.after(0, lambda: self._log(msg, color))
        def status(msg):
            self.after(0, lambda: self.v_status.set(msg))
        def progress(v):
            self.after(0, lambda: self.pb.configure(value=v))

        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            images = sorted(
                p for p in Path(input_dir).iterdir()
                if p.is_file() and p.suffix.lower() in EXTS
            )
            total = len(images)
            log(f"พบรูปทั้งหมด: {total} ไฟล์")
            self.after(0, lambda: self.pb.configure(maximum=max(total, 1)))

            # ── Face detector ──────────────────────────────────
            if os.path.exists(FACE_MODEL):
                detector = cv2.FaceDetectorYN.create(
                    FACE_MODEL, "", (320, 320),
                    score_threshold=0.6, nms_threshold=0.3
                )
                log("Face detector: YuNet ✓", "#7ecf7e")
            else:
                detector = None
                log("YuNet ไม่พบ → Haar cascade fallback", "#f0c040")

            haar_xml = HAAR_PATH if os.path.exists(HAAR_PATH) else (
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            cascade = cv2.CascadeClassifier(haar_xml)

            # ── AI Upscaler ────────────────────────────────────
            upscaler = None
            if use_up:
                if self._upscaler is None:
                    log("กำลังโหลด ONNX upscale model…", "#a0c8ff")
                    self._upscaler = OnnxUpscaler(up_model_path)
                upscaler = self._upscaler
                log(f"Upscaler: {Path(up_model_path).name}  "
                    f"x{upscaler.scale}  [{upscaler.device}]", "#7ecf7e")

            log("─" * 54)
            ext_map = {"JPEG": ".jpg", "PNG": ".png"}
            ok, skipped = 0, []

            for i, img_path in enumerate(images):
                if not self._running:
                    log("— หยุดโดยผู้ใช้ —", "#f0c040")
                    break

                progress(i + 1)
                status(f"[{i+1}/{total}]  {img_path.name}")

                img = cv2.imread(str(img_path))
                if img is None:
                    log(f"SKIP  {img_path.name}  (อ่านไม่ได้)", "#ff7070")
                    skipped.append(img_path.name); continue

                h, w = img.shape[:2]
                face = None

                # YuNet
                if detector is not None:
                    detector.setInputSize((w, h))
                    _, faces = detector.detect(img)
                    if faces is not None and len(faces) > 0:
                        b = faces[faces[:, 14].argmax()]
                        face = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

                # Haar fallback
                if face is None:
                    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    found = cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                    )
                    if len(found) == 0:
                        log(f"SKIP  {img_path.name}  (ไม่พบหน้า)", "#ff9040")
                        skipped.append(img_path.name); continue
                    x, y, fw, fh = max(found, key=lambda r: r[2] * r[3])
                    face = (float(x), float(y), float(fw), float(fh))

                # Crop
                bx, by, bw, bh = face
                cx = bx + bw / 2;  cy = by + bh / 2
                sz = max(bw, bh) * padding
                x1 = max(0, int(cx - sz/2));  y1 = max(0, int(cy - sz/2))
                x2 = min(w, int(cx + sz/2));  y2 = min(h, int(cy + sz/2))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    log(f"SKIP  {img_path.name}  (crop ว่าง)", "#ff7070")
                    skipped.append(img_path.name); continue

                ch, cw = crop.shape[:2]
                up_note = ""

                # ── AI Upscale ──────────────────────────────────
                if upscaler is not None:
                    need_up = (up_thresh == 0) or (min(ch, cw) < up_thresh)
                    if need_up:
                        crop    = upscaler.upscale(crop)
                        ch, cw  = crop.shape[:2]
                        up_note = f" [↑{upscaler.scale}x]"

                # Resize to max_size (keep aspect ratio)
                scale = max_size / max(cw, ch)
                new_w = int(round(cw * scale))
                new_h = int(round(ch * scale))

                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                pil = pil.resize((new_w, new_h), Image.LANCZOS)

                out_name = img_path.stem + ext_map[fmt]
                out_path = Path(output_dir) / out_name
                if fmt == "JPEG":
                    pil.save(str(out_path), "JPEG", quality=quality, subsampling=0)
                else:
                    pil.save(str(out_path), "PNG")

                ok += 1
                log(f"OK  {img_path.name}  →  {new_w}×{new_h}{up_note}")

            log("─" * 54)
            log(f"เสร็จ: {ok} / {total} รูป", "#7ecf7e")
            if skipped:
                log(f"ข้ามไป {len(skipped)} รูป:", "#f0c040")
                for s in skipped:
                    log(f"   • {s}", "#f0c040")
            status(f"Done: {ok}/{total}")
            self.after(0, lambda: self.btn_open.configure(state="normal"))

        except Exception:
            import traceback
            self.after(0, lambda: self._log(traceback.format_exc(), "#ff7070"))
        finally:
            self._running = False
            self.after(0, lambda: self.btn_start.configure(state="normal"))
            self.after(0, lambda: self.btn_stop.configure(state="disabled"))


if __name__ == "__main__":
    App().mainloop()
