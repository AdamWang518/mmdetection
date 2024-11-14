import torch

# 檢查 CUDA 是否可用
print("CUDA is available:", torch.cuda.is_available())

# 如果 CUDA 可用，檢查 CUDA 版本和 GPU 信息
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)
    print("Number of GPUs available:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU detected.")
