import torch
import gymnasium
import metaworld

print(f"Python Version: 3.10 (Safe for RL)")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CPU Mode detected.")

print("Gymnasium and MetaWorld imported successfully.")