import torch
p=r"E:\chronos\models\latest.pth"
ckpt=torch.load(p, map_location="cpu")
print("keys:", list(ckpt.keys())[:10] if isinstance(ckpt, dict) else type(ckpt))
