from __future__ import annotations
import struct
from pathlib import Path
import torch
from .model import MLPValue

MAGIC = b"CHNN"
VERSION = 1

# nn_sl.bin format:
# 4 bytes MAGIC
# uint32 version
# uint32 input_dim
# uint32 h1,h2,h3
# then each layer weights row-major float32 + biases float32
# layer1: (h1,input_dim), b1(h1)
# layer2: (h2,h1), b2(h2)
# layer3: (h3,h2), b3(h3)
# layer4: (1,h3), b4(1)

def export_mlp(model: MLPValue, out_path: Path):
    model.eval()
    layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    assert len(layers) == 4

    w1, b1 = layers[0].weight.detach().cpu().numpy(), layers[0].bias.detach().cpu().numpy()
    w2, b2 = layers[1].weight.detach().cpu().numpy(), layers[1].bias.detach().cpu().numpy()
    w3, b3 = layers[2].weight.detach().cpu().numpy(), layers[2].bias.detach().cpu().numpy()
    w4, b4 = layers[3].weight.detach().cpu().numpy(), layers[3].bias.detach().cpu().numpy()

    input_dim = w1.shape[1]
    h1 = w1.shape[0]
    h2 = w2.shape[0]
    h3 = w3.shape[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIIIII", VERSION, input_dim, h1, h2, h3, 0))  # reserved=0
        for arr in (w1, b1, w2, b2, w3, b3, w4, b4):
            f.write(arr.astype("float32").tobytes(order="C"))
