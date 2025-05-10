import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((7,)),  # Only one arm's state (6 joints + 1 gripper)
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [7] (one arm's state - 6 joints + 1 gripper)
    - actions: [action_horizon, 7]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_wrist")

    def __call__(self, data: dict) -> dict:
        # Duplicate single arm state to create bimanual state
        single_arm_state = np.asarray(data["state"])
        
        # Create a full 14-dim state by duplicating the single arm state
        bimanual_state = np.zeros(14)
        # First 6 joints for left arm
        bimanual_state[:6] = single_arm_state[:6]
        # Left gripper
        bimanual_state[6] = single_arm_state[6]
        # Right arm gets same joint values
        bimanual_state[7:13] = single_arm_state[:6]
        # Right gripper gets same value
        bimanual_state[13] = single_arm_state[6]
        
        # Use the existing decoding with our constructed bimanual state
        data_bimanual = data.copy()
        data_bimanual["state"] = bimanual_state
        data_bimanual = _decode_aloha(data_bimanual, adapt_to_pi=self.adapt_to_pi)

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data_bimanual["state"], self.action_dim)

        in_images = data_bimanual["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images, duplicating the single wrist camera for both arms
        wrist_img = in_images.get("cam_wrist", None)
        if wrist_img is not None:
            images["left_wrist_0_rgb"] = wrist_img
            images["right_wrist_0_rgb"] = wrist_img
            image_masks["left_wrist_0_rgb"] = np.True_
            image_masks["right_wrist_0_rgb"] = np.True_
        else:
            images["left_wrist_0_rgb"] = np.zeros_like(base_image)
            images["right_wrist_0_rgb"] = np.zeros_like(base_image)
            image_masks["left_wrist_0_rgb"] = np.False_
            image_masks["right_wrist_0_rgb"] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            # Duplicate single arm actions to create bimanual actions
            single_arm_actions = np.asarray(data["actions"])
            action_horizon = single_arm_actions.shape[0]
            
            # Create bimanual actions by duplicating for both arms
            bimanual_actions = np.zeros((action_horizon, 14))
            for i in range(action_horizon):
                # First 6 joints + gripper for left arm
                bimanual_actions[i, :6] = single_arm_actions[i, :6]
                bimanual_actions[i, 6] = single_arm_actions[i, 6]
                # Same joints + gripper for right arm
                bimanual_actions[i, 7:13] = single_arm_actions[i, :6]
                bimanual_actions[i, 13] = single_arm_actions[i, 6]
            
            actions = _encode_actions_inv(bimanual_actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Get the full bimanual actions but only return the left arm actions (first 7 dims)
        bimanual_actions = np.asarray(data["actions"][:, :14])
        bimanual_actions = _encode_actions(bimanual_actions, adapt_to_pi=self.adapt_to_pi)
        
        # Extract only the left arm actions (first 7 dims - 6 joints + 1 gripper)
        single_arm_actions = np.zeros((bimanual_actions.shape[0], 7))
        single_arm_actions[:, :6] = bimanual_actions[:, :6]  # First 6 joints
        single_arm_actions[:, 6] = bimanual_actions[:, 6]    # Gripper
        
        return {"actions": single_arm_actions}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = _unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
