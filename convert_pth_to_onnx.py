"""
แปลง RealESRGAN_x4plus.pth → realesrgan_x4.onnx
ไม่ต้องติดตั้ง basicsr/realesrgan เพราะเขียน architecture ไว้ตรงนี้เลย
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path


# ─── RRDB architecture (Real-ESRGAN x4plus) ─────────────────────────────────

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4,
                 num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        num_up = int(scale ** 0.5)  # 2 for x4

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body = self.conv_body(self.body(feat))
        feat = feat + body
        feat = self.lrelu(self.conv_up1(nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
        out  = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ─── main ────────────────────────────────────────────────────────────────────

def convert(pth_path: str, out_path: str):
    print(f"โหลด: {pth_path}")
    sd = torch.load(pth_path, map_location="cpu", weights_only=True)

    # state dict อาจซ้อนอยู่ใน key 'params_ema' หรือ 'params'
    if "params_ema" in sd:
        sd = sd["params_ema"]
    elif "params" in sd:
        sd = sd["params"]

    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4,
                    num_feat=64, num_block=23, num_grow_ch=32)
    model.load_state_dict(sd, strict=True)
    model.eval()

    dummy = torch.zeros(1, 3, 64, 64)
    print("Export ONNX ...")
    torch.onnx.export(
        model, dummy, out_path,
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "h", 3: "w"},
                      "output": {0: "batch", 2: "h", 3: "w"}},
    )
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"บันทึก: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    pth = sys.argv[1] if len(sys.argv) > 1 else r"D:\Downloads\RealESRGAN_x4plus.pth"
    out = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / "realesrgan_x4.onnx")
    convert(pth, out)
