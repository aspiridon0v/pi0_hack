import dataclasses
import os

import jax

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

if os.environ.get("BV_JAX_INIT"):
    jax.distributed.initialize()

config = _config.get_config("pi0_droid")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = droid_policy.make_droid_example()
result = policy.infer(example)

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)