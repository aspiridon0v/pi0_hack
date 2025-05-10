import os

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import rerun as rr

import openpi.training.config as _config

os.environ["LEROBOT_HOME"] = "/mnt/disks/persist/hf_cache"
os.environ["HF_HOME"] = "/mnt/disks/persist/hf_cache"


def main(config: _config.TrainConfig):
    repo_id = "first/piper_dataset"
    episode_index = 1

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
        repo_id, local_files_only=True
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id=repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(50)]
            for key in data_config.action_sequence_keys
        },
        local_files_only=data_config.local_files_only,
    )
    episode_start_index = dataset.episode_data_index["from"][episode_index]
    episode_end_index = dataset.episode_data_index["to"][episode_index]

    rr.init("rerun_example_points3d", spawn=True)
    rr.connect_tcp("127.0.0.1:9876")

    subsampling_rate = 30
    time_indices = np.arange(episode_start_index, episode_end_index, subsampling_rate)

    for index, t in enumerate(time_indices.tolist()):
        print(index)
        rr.set_time_seconds("time", t * (1 / 50) * subsampling_rate)

        episode_data = dataset[t]
        image = episode_data["image"]
        image = np.array(image) * 255
        image = image.astype(np.uint8).transpose(1, 2, 0)
        rr.log("image", rr.Image(image))

        wrist_image = episode_data["wrist_image"]
        wrist_image = np.array(wrist_image) * 255
        wrist_image = wrist_image.astype(np.uint8).transpose(1, 2, 0)
        rr.log("wrist_image", rr.Image(wrist_image))

        state = np.array(episode_data["state"])  # (7)
        for i, val in enumerate(state):
            rr.log(f"state/dof_{i}", rr.Scalar(val))

    # rr.log(
    #     "/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True
    # )  # Set the global coordinate system

    # image = np.array(episode_data["image"]) * 255
    # image = image.astype(np.uint8).transpose(1, 2, 0)

    # wrist_image = np.array(episode_data["wrist_image"]) * 255
    # wrist_image = wrist_image.astype(np.uint8).transpose(1, 2, 0)

    # rr.set_time_seconds("time", 0)
    # rr.log("image", rr.Image(image))
    # rr.log("wrist_image", rr.Image(wrist_image))
    # for t in range(video.shape[0]):
    #     rr.set_time_seconds("time", t * (1 / 50))

    #     image = np.array(video[t]["image"]) * 255
    #     image = image.astype(np.uint8).transpose(1, 2, 0)
    #     rr.log("image", rr.Image(image))


if __name__ == "__main__":
    main(_config.cli())
