import tensorflow as tf
import sys
import os

# Suppress informational messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print("--- Keras/TensorFlow GPU Compatibility Check ---")
print(f"Using TensorFlow version: {tf.__version__}")
print(f"Using Python version: {sys.version.split()[0]}")

try:
    build_info = tf.sysconfig.get_build_info()
    print(f"TensorFlow was built with CUDA version: {build_info.get('cuda_version', 'N/A')}")
    print(f"TensorFlow was built with cuDNN version: {build_info.get('cudnn_version', 'N/A')}")
except (KeyError, AttributeError):
    print("Could not retrieve TensorFlow build info. This might happen on older versions.")

print("\n--- Checking for GPU availability ---")

try:
    # List physical devices, which forces initialization
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.name}")

        print("\nSUCCESS: TensorFlow can see and initialize the GPU.")
        print("\n--- Simple GPU computation test ---")
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                c = tf.matmul(a, b)
            print("Matrix multiplication on GPU successful.")
        except Exception as e:
            print(f"\nWARNING: Could not perform a test computation on the GPU. Error: {e}")
    else:
        print("\nError: TensorFlow did not find any available GPUs.")
        
except Exception as e:
    print(f"\nAn error occurred during TensorFlow initialization: {e}")