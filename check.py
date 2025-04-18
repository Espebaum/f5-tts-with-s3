from safetensors.torch import load_file

ckpt_path = "/mnt/e/home/gyopark/.cache/huggingface/hub/models--SWivid--F5-TTS/snapshots/d6bd6c3c3ec65c0a3ef25a6d3d09658c5e2817fd/F5TTS_v1_Base/model_1250000.safetensors"

state_dict = load_file(ckpt_path)

# EMA 버전 확인
ema_key = "ema_model.transformer.text_embed.text_embed.weight"
regular_key = "transformer.text_embed.text_embed.weight"

if ema_key in state_dict:
    vocab_size, dim = state_dict[ema_key].shape
    print(f"✅ EMA vocab size: {vocab_size}, embedding dim: {dim}")
elif regular_key in state_dict:
    vocab_size, dim = state_dict[regular_key].shape
    print(f"✅ Regular vocab size: {vocab_size}, embedding dim: {dim}")
else:
    print("❌ text_embed.weight key not found in checkpoint.")
