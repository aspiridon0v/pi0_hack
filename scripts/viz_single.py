import os

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import rerun as rr

import openpi.training.config as _config

os.environ["LEROBOT_HOME"] = "/mnt/disks/persist/hf_cache"
os.environ["HF_HOME"] = "/mnt/disks/persist/hf_cache"


def main(config: _config.TrainConfig):
    repo_id = "first/piper_dataset"

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

    rr.init("rerun_example_points3d", spawn=True)
    rr.connect_tcp("127.0.0.1:9876")

    sample_index = 15

    episode_data = dataset[sample_index]
    image = episode_data["image"]
    image = np.array(image) * 255
    image = image.astype(np.uint8).transpose(1, 2, 0)
    rr.log("image", rr.Image(image))

    wrist_image = episode_data["wrist_image"]
    wrist_image = np.array(wrist_image) * 255
    wrist_image = wrist_image.astype(np.uint8).transpose(1, 2, 0)

    action = np.array(episode_data["actions"])  # (7)
    subsampling_rate = 30
    time_indices = np.arange(0, action.shape[0], subsampling_rate)
    for t in time_indices.tolist():
        rr.set_time_seconds("time", t * (1 / 50))
        rr.log("image", rr.Image(image))
        rr.log("wrist_image", rr.Image(wrist_image))
        for i, val in enumerate(action[t]):
            rr.log(f"action/dof_{i}", rr.Scalar(val.item()))


if __name__ == "__main__":
    main(_config.cli())
