import os

import jax

# Set TPU configuration before importing JAX operations
os.environ['PJRT_DEVICE'] = 'TPU'

print("JAX version:", jax.__version__)

# Print environment variables related to TPU configuration
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS", "Not set"))
print("PJRT_DEVICE:", os.environ.get("PJRT_DEVICE", "Not set"))
print("TPU_NAME:", os.environ.get("TPU_NAME", "Not set"))

# Try to explicitly initialize TPU
try:
    print("Attempting to initialize TPU...")
    jax.config.update('jax_platform_name', 'tpu')
    print("Available devices:", jax.devices())
except Exception as e:
    print("TPU initialization error:", repr(e))

# Try to get device count
try:
    print("TPU device count:", jax.device_count())
except Exception as e:
    print("Error getting device count:", repr(e))

# Check if TPU is available
try:
    from jax.lib import xla_bridge
    print("Default backend:", xla_bridge.get_backend().platform)
except Exception as e:
    print("Error getting backend:", repr(e))