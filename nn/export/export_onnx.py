from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nn.models.chronos_cnn import ChronosCNN
from nn.models.chronos_hybrid import ChronosHybridNet
from nn.move_index import MOVE_SPACE


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, with_policy: bool):
        super().__init__()
        self.model = model
        self.with_policy = with_policy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        scalars = torch.cat([out["value"], out["pressure"], out["volatility"], out["complexity"]], dim=1)  # [B,4]
        if self.with_policy:
            return torch.cat([scalars, out["policy"]], dim=1)  # [B, 4+MOVE_SPACE]
        return scalars


def load_model(ckpt: dict) -> tuple[torch.nn.Module, bool]:
    kind = str(ckpt.get("kind", "cnn4"))
    if kind.startswith("hybrid"):
        m = ChronosHybridNet(in_planes=25)
        m.load_state_dict(ckpt["model"], strict=True)
        return m, True
    m = ChronosCNN(in_planes=25)
    m.load_state_dict(ckpt["model"], strict=True)
    return m, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--out", default="", help="Output ONNX path (default: same folder / model.onnx)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model, with_policy = load_model(ckpt)
    model.eval()
    wrapper = ExportWrapper(model, with_policy=with_policy).eval()

    out_path = Path(args.out) if args.out else (ckpt_path.parent / ("chronos_hybrid.onnx" if with_policy else "chronos.onnx"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros((1, 25, 8, 8), dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=int(args.opset),
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    print(f"Exported: {out_path}")
    if with_policy:
        print(f"Output: [B, 4+{MOVE_SPACE}] (scalars + policy logits)")
    else:
        print("Output: [B, 4] (value, pressure, volatility, complexity)")


if __name__ == "__main__":
    main()
