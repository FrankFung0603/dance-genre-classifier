"""Spike 3a: 匯出 torchvision R(2+1)D-18（Kinetics-400 預訓練）為 ONNX（一次性）。

輸入 shape: (1, 3, 16, 224, 224) = 16 frames @ 224x224
輸出: models/r2plus1d_18_k400.onnx

注意：CPU-only torch 即可匯出；權重約 130MB，首次執行會下載。
"""
from pathlib import Path

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "r2plus1d_18_k400.onnx"


def main() -> None:
    print("loading R(2+1)D-18 KINETICS400_V1 weights ...")
    model = torchvision.models.video.r2plus1d_18(
        weights=torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1
    )
    model.eval()

    dummy = torch.randn(1, 3, 16, 224, 224)
    print("exporting to ONNX ...")
    torch.onnx.export(
        model,
        dummy,
        str(OUT),
        input_names=["frames"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = OUT.stat().st_size / 1e6
    print(f"saved: {OUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
