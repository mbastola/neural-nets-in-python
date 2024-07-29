import torch
import sys

print("--- PyTorch GPU Compatibility Check ---")
print(f"Using PyTorch version: {torch.__version__}")
print(f"Using Python version: {sys.version.split()[0]}")

if torch.cuda.is_available():
    print(f"Found {torch.cuda.device_count()} GPU(s) available.")

    # PyTorch CUDA and cuDNN versions
    print(f"PyTorch was built with CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

    print("\nSUCCESS: PyTorch can see and use the GPU.")

    # Run a simple computation on the GPU
    try:
        print("\n--- Simple GPU computation test ---")
        device = torch.device("cuda")
        a = torch.randn(100, 100).to(device)
        b = torch.randn(100, 100).to(device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize() # Wait for the operation to complete
        print("Matrix multiplication on GPU successful.")
    except Exception as e:
        print(f"\nWARNING: Could not perform a test computation on the GPU. Error: {e}")

else:
    print("\nError: PyTorch did not find any available GPUs.")