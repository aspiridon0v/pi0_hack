import fnmatch
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

# from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main(data_directory, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = Path("/mnt/disks/persist/piper_dataset")
    if output_path.exists():
        shutil.rmtree(output_path)
    # Create LeRobot dataset, define features to store
    dataset = LeRobotDataset.create(
        repo_id="",
        root=output_path,
        robot_type=None,
        fps=50,  # Check FPS from data
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),  # Check shape before/after transformation.
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),  # Same check as above
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over episodes as each folder in the data directory.
    for episode in sorted(os.listdir(data_directory)):
        episode_path = os.path.join(data_directory, episode)
        if int(episode.split("_")[-1]) <= 10:
            task = "pick up orange can"
        else:
            task = "pick up blue can"
        # Assert that the number of images in the folder wrist_images and the folder images are the same.
        wrist_images_path = os.path.join(episode_path, "wrist")
        images_path = os.path.join(episode_path, "exterior")
        assert len(os.listdir(wrist_images_path)) == len(
            os.listdir(images_path)
        ), "Number of images in wrist_images and images folder are not the same."
        # Loop over both wrist_images folder and images folder and load all images to a list.
        wrist_image_paths = dict(
            [
                (int(file.split("_")[-1][:-4]), file)
                for file in os.listdir(wrist_images_path)
            ]
        )
        image_paths = dict(
            [(int(file.split("_")[-1][:-4]), file) for file in os.listdir(images_path)]
        )
        sorted_wrist_keys = sorted(wrist_image_paths.keys())
        sorted_image_keys = sorted(image_paths.keys())
        images = []
        wrist_images = []
        for wrist_key, image_key in zip(sorted_wrist_keys, sorted_image_keys):
            wrist_image_path = os.path.join(
                wrist_images_path, wrist_image_paths[wrist_key]
            )
            image_path = os.path.join(images_path, image_paths[image_key])
            wrist_image = cv2.cvtColor(cv2.imread(wrist_image_path), cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            wrist_images.append(wrist_image)
            images.append(image)

        # Read the joint_angles .npz file by finding file with joint_angles in filename.
        joint_angles = os.path.join(episode_path, f"joint_angles_{episode}.npz")
        gripper_pos = os.path.join(episode_path, f"gripper_state_{episode}.npz")
        # Read the numpy array from the .npz file.
        joint_angles_array = np.load(joint_angles)["joint_angles"]
        gripper_pos_array = np.load(gripper_pos)["gripper_state"]
        state_trajectory_full = np.concatenate(
            (joint_angles_array, gripper_pos_array.reshape(-1, 1)), axis=1
        )
        # Get the delta trajectory from the numpy array.
        action_trajectory = state_trajectory_full[1:, :]
        state_trajectory = state_trajectory_full[:-1, :]
        len_episode = len(state_trajectory)
        assert (
            len_episode == len(images) == len(wrist_images) == len(action_trajectory)
        ), "Length of episode, images, wrist_images and action_trajectory are not the same."
        # Loop over the images and wrist_images and add them to the dataset.
        for i in range(len_episode):
            # Get images and state
            image = images[i]
            wrist_image = wrist_images[i]
            state = state_trajectory[i].reshape(-1)
            actions = action_trajectory[i].reshape(-1)

            # Add data to dataset
            dataset.add_frame(
                {
                    "image": image,
                    "wrist_image": wrist_image,
                    "state": state,
                    "actions": actions,
                }
            )

        # Save the episode to the dataset.
        dataset.save_episode(task=task)

    # Consolidate the dataset.
    dataset.consolidate(run_compute_stats=False)
    # # Push the dataset to the hub.
    # if push_to_hub:
    #     dataset.push_to_hub(
    #         repo_id="",
    #         commit_message="Initial commit",
    #         private=True,
    #         organization=None,
    #         token=None,
    #     )


def main_dummy(data, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = Path("dummy_dataset")
    if output_path.exists():
        shutil.rmtree(output_path)
    # Create LeRobot dataset, define features to store
    dataset = LeRobotDataset.create(
        repo_id="",
        root="ADD_DATASET_PATH_HERE",
        robot_type=None,
        fps=10,  # Check FPS from data
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),  # Check shape before/after transformation.
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),  # Same check as above
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over custom dataset format to fill up the LeRobot dataset
    for step in data:
        # Get images and state
        image = step["observation/exterior_image_1_left"]
        wrist_image = step["observation/wrist_image_left"]
        state = step["observation/joint_position"]
        actions = step["action"]

        # Add data to dataset
        dataset.add_frame(
            {
                "image": image,
                "wrist_image": wrist_image,
                "state": state,
                "actions": actions,
            }
        )
    dataset.save_episode(task=step["prompt"])
    # Consolidate the dataset.
    dataset.consolidate(run_compute_stats=False)


def create_dummy_dataset():
    # Create a dummy list of dicts filled with data to convert to LeRobot dataset
    data = []
    for i in range(100):
        data.append(
            {
                "observation/exterior_image_1_left": np.random.randint(
                    256, size=(256, 256, 3), dtype=np.uint8
                ),
                "observation/wrist_image_left": np.random.randint(
                    256, size=(256, 256, 3), dtype=np.uint8
                ),
                "observation/joint_position": np.random.rand(7),
                "action": np.random.rand(7),
                "prompt": "do something",
            }
        )
    return data


if __name__ == "__main__":
    # Create dummy dataset
    data = Path("/home/alex/dataset")
    # Convert to LeRobot dataset
    main(data, push_to_hub=False)
    # data = Path("/home/alex/dataset")
    # output_path = Path("/mnt/disks/persist/piper_dataset")
