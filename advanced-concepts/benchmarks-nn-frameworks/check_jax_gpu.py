import jax
import jax.numpy as jnp
import sys

print("--- JAX GPU Compatibility Check ---")
print(f"Using JAX version: {jax.__version__}")
print(f"Using Python version: {sys.version.split()[0]}")

try:
    print(f"\nDefault JAX backend: {jax.default_backend()}")
    devices = jax.devices()
    print(f"Available JAX devices: {devices}")

    gpu_devices = [d for d in devices if d.platform == 'gpu']

    if gpu_devices:
        print(f"\nFound {len(gpu_devices)} JAX GPU device(s).")
        print("\n✅ SUCCESS: JAX is configured to use the GPU.")

        # Run a simple computation on the GPU
        try:
            key = jax.random.PRNGKey(0)
            x = jax.random.normal(key, (100, 100))
            # Use jax.device_put to ensure it's on the first GPU
            x_on_gpu = jax.device_put(x, gpu_devices[0])
            y = jnp.dot(x_on_gpu, x_on_gpu).block_until_ready()
            print("\n--- Simple GPU computation test ---")
            print("Matrix multiplication on GPU successful.")
            print("\n✅ Your environment appears to be correctly configured for JAX with GPU support.")
        except Exception as e:
            print(f"\n⚠️ WARNING: Could not perform a test computation on the GPU. Error: {e}")

    else:
        print("\n❌ FAILURE: JAX did not find any available GPUs.")
        print("Please check the following:")
        print("  1. NVIDIA GPU drivers are installed correctly (check with `nvidia-smi`).")
        print("  2. You have installed a CUDA-enabled version of `jaxlib`.")
        print("     Example installation command for CUDA 12:")
        print("     pip install --upgrade \"jax[cuda12_pip]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

except Exception as e:
    print(f"\n❌ An error occurred during JAX initialization: {e}")
