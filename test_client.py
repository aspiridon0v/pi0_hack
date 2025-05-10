import numpy as np
from openpi_client import image_tools, websocket_client_policy

from openpi.policies import droid_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
example = droid_policy.make_droid_example()
for step in range(10):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(example["observation/exterior_image_1_left"], 224, 224)
        ),
        "observation/wrist_image_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(example["observation/wrist_image_left"], 224, 224)
        ),
        "observation/joint_position": example["observation/joint_position"],
        "observation/gripper_position": example["observation/gripper_position"],
        "prompt": "do something",
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]
    print(action_chunk.shape)
