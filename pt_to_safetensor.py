import torch
from safetensors.torch import save_file

# 경로 설정
pt_path = "/mnt/e/home/gyopark/F5-TTS/ckpts/emilia_l40/model_2.pt"
safetensors_path = "/mnt/e/home/gyopark/F5-TTS/ckpts/emilia_l40/model_2.safetensors"

# 🔄 체크포인트 로드
ckpt = torch.load(pt_path, map_location="cpu")

# 🔍 state_dict 형태로 되어 있는지 확인
if "model_state_dict" in ckpt:
    tensor_dict = ckpt["model_state_dict"]
elif "ema_model" in ckpt:
    tensor_dict = ckpt["ema_model"]
else:
    tensor_dict = ckpt  # 바로 state_dict일 수도 있음

# 💾 safetensors로 저장
save_file(tensor_dict, safetensors_path)

print(f"✅ Converted: {pt_path} → {safetensors_path}")
