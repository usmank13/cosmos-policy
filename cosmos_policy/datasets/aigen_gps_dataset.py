# Aigen GPS Driving Dataset for Cosmos Policy
#
# Loads HDF5 files from GPS-guided driving sessions.
# Each HDF5 file = one episode with:
#   - images: front nav camera frames (T, H, W, 3) uint8
#   - proprio: (T, 5) float32 [x, y, theta, v, omega]
#   - actions: (T, 2) float32 [linear_vel, angular_vel]
#   - command: str (task description, e.g. "follow the crop rows")

import os
import pickle
import time

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.dataset_common import (
    build_demo_step_index_mapping,
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    determine_sample_type,
    get_action_chunk_with_padding,
    load_or_compute_dataset_statistics,
)
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    get_hdf5_files,
    preprocess_image,
    rescale_data,
    resize_images,
)

np.set_printoptions(precision=3, linewidth=np.inf)


class AigenGPSDrivingDataset(Dataset):
    """
    Dataset for Aigen robot GPS-guided driving sessions.

    Simpler than ALOHA: single camera, 2-DOF actions, 5-dim proprio.

    Latent structure (state_t=7):
        0: blank
        1: proprio (4 latent frames)
        2: primary image (4 latent frames)
        3: action (4 latent frames)
        4: future proprio (4 latent frames)
        5: future primary image (4 latent frames)
        6: value (4 latent frames)
    Total chunk_duration = 1 + 6*4 = 25
    """

    def __init__(
        self,
        data_dir: str,
        is_train: bool = True,
        chunk_size: int = 50,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        debug: bool = False,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        return_value_function_returns: bool = False,
        gamma: float = 0.99,
        demonstration_sampling_prob: float = 0.5,
        success_rollout_sampling_prob: float = 0.5,
        treat_demos_as_success_rollouts: bool = True,
        # Unused kwargs for config compatibility
        use_wrist_images: bool = False,
        use_third_person_images: bool = False,
        rollout_data_dir: str = "",
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.debug = debug
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.success_rollout_sampling_prob = success_rollout_sampling_prob

        # Load HDF5 files
        hdf5_files = get_hdf5_files(data_dir, is_train=is_train)
        if debug:
            hdf5_files = hdf5_files[:1]

        # Load all episodes into RAM
        self.data = {}
        self.num_episodes = 0
        self.num_steps = 0
        self.unique_commands = set()

        for file in tqdm(hdf5_files, desc="Loading Aigen GPS episodes"):
            with h5py.File(file, "r") as f:
                # Load images
                if "images" in f:
                    images = f["images"][:]  # (T, H, W, 3) uint8
                elif "observations/images/front_nav" in f:
                    images = f["observations/images/front_nav"][:]
                else:
                    raise KeyError(f"No images found in {file}. Expected 'images' or 'observations/images/front_nav'")

                # Resize if needed
                if images.shape[1] != final_image_size or images.shape[2] != final_image_size:
                    images = resize_images(images, final_image_size)

                # Load actions
                actions = f["actions"][:]  # (T, 2) [linear_vel, angular_vel]

                # Load proprio
                proprio = f["proprio"][:]  # (T, 5) [x, y, theta, v, omega]

                # Load command/task description
                if "command" in f.attrs:
                    command = f.attrs["command"]
                elif "command" in f:
                    command = f["command"][()].decode("utf-8") if isinstance(f["command"][()], bytes) else str(f["command"][()])
                else:
                    command = "follow the crop rows"

                num_steps = len(images)
                assert len(actions) == num_steps, f"actions length mismatch in {file}"
                assert len(proprio) == num_steps, f"proprio length mismatch in {file}"

                episode = {
                    "images": images,
                    "proprio": proprio,
                    "actions": actions,
                    "command": command,
                    "num_steps": num_steps,
                }

                # Compute returns for value function
                if return_value_function_returns:
                    # Treat each episode as successful (terminal_reward=1)
                    returns = compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=gamma)
                    episode["returns"] = returns

                self.data[self.num_episodes] = episode
                self.num_episodes += 1
                self.num_steps += num_steps
                self.unique_commands.add(command)

        print(f"Loaded {self.num_episodes} episodes, {self.num_steps} total steps")
        print(f"Unique commands: {self.unique_commands}")

        # Compute dataset statistics
        self.dataset_stats = load_or_compute_dataset_statistics(
            data_dir=data_dir,
            data=self.data,
            calculate_dataset_statistics_func=calculate_dataset_statistics,
        )

        # Normalize actions and proprio
        if normalize_actions:
            self.data = rescale_data(self.data, self.dataset_stats, "actions")
        if normalize_proprio:
            self.data = rescale_data(self.data, self.dataset_stats, "proprio")

        # Build step-to-episode mapping
        mapping_result = build_demo_step_index_mapping(self.data)
        self._step_to_episode_map = mapping_result["_step_to_episode_map"]

        # Epoch structure (for demo/rollout sampling)
        epoch_info = calculate_epoch_structure(
            num_steps=self.num_steps,
            rollout_success_total_steps=self.num_steps if treat_demos_as_success_rollouts else 0,
            rollout_failure_total_steps=0,
            demonstration_sampling_prob=demonstration_sampling_prob,
            success_rollout_sampling_prob=success_rollout_sampling_prob,
        )
        self.adjusted_demo_count = epoch_info["adjusted_demo_count"]
        self.adjusted_success_rollout_count = epoch_info["adjusted_success_rollout_count"]
        self.adjusted_failure_rollout_count = epoch_info["adjusted_failure_rollout_count"]

        # If treating demos as success rollouts, reuse the same step map
        if treat_demos_as_success_rollouts:
            self._rollout_success_step_to_episode_map = self._step_to_episode_map
            self._rollout_success_total_steps = self.num_steps

        # Load T5 text embeddings
        if t5_text_embeddings_path and os.path.exists(t5_text_embeddings_path):
            with open(t5_text_embeddings_path, "rb") as f:
                self.t5_text_embeddings = pickle.load(f)
            print(f"Loaded T5 embeddings for {len(self.t5_text_embeddings)} commands")
        else:
            print("WARNING: No T5 text embeddings loaded. Will use zeros.")
            self.t5_text_embeddings = {}

    def __len__(self):
        return self.adjusted_demo_count + self.adjusted_success_rollout_count + self.adjusted_failure_rollout_count

    def __getitem__(self, idx):
        if self.debug:
            idx = 0

        # Determine sample type
        sample_type = determine_sample_type(idx, self.adjusted_demo_count, self.adjusted_success_rollout_count)

        if sample_type == "demo":
            global_step_idx = idx % self.num_steps
            episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
        elif sample_type == "success_rollout":
            success_idx = idx - self.adjusted_demo_count
            global_rollout_idx = success_idx % max(1, getattr(self, "_rollout_success_total_steps", 1))
            episode_idx, relative_step_idx = self._rollout_success_step_to_episode_map[global_rollout_idx]
        else:
            # Fallback to demo data
            global_step_idx = idx % self.num_steps
            episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]

        episode_data = self.data[episode_idx]

        # Calculate future frame index
        future_frame_idx = min(
            relative_step_idx + self.chunk_size,
            episode_data["num_steps"] - 1,
        )

        # Get proprio
        proprio = episode_data["proprio"][relative_step_idx]
        future_proprio = episode_data["proprio"][future_frame_idx]

        # Get images
        current_image = episode_data["images"][relative_step_idx]
        future_image = episode_data["images"][future_frame_idx]

        # Build frame sequence for latent diffusion
        # Structure: blank, [proprio], primary, action, [future_proprio], future_primary, [value]
        frames = []
        repeats = []
        segment_idx = 0

        # Pre-init latent indices
        current_proprio_latent_idx = -1
        current_image_latent_idx = -1
        action_latent_idx = -1
        future_proprio_latent_idx = -1
        future_image_latent_idx = -1
        value_latent_idx = -1

        # Blank placeholder indices for wrist cameras (we don't have them)
        current_wrist_image_latent_idx = -1
        current_wrist_image2_latent_idx = -1
        future_wrist_image_latent_idx = -1
        future_wrist_image2_latent_idx = -1

        ref_shape = current_image
        blank = np.zeros_like(ref_shape)

        # 0: Blank first frame
        frames.append(blank)
        repeats.append(1)
        segment_idx += 1

        # 1: Current proprio (blank image, values injected in latent space)
        if self.use_proprio:
            current_proprio_latent_idx = segment_idx
            frames.append(blank.copy())
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

        # 2: Current primary image
        current_image_latent_idx = segment_idx
        frames.append(current_image)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # 3: Action chunk (blank image, actions in latent space)
        action_latent_idx = segment_idx
        frames.append(blank.copy())
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # 4: Future proprio
        if self.use_proprio:
            future_proprio_latent_idx = segment_idx
            frames.append(blank.copy())
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

        # 5: Future primary image
        future_image_latent_idx = segment_idx
        frames.append(future_image)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # 6: Value (blank image)
        if self.return_value_function_returns:
            value_latent_idx = segment_idx
            frames.append(blank.copy())
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

        # Preprocess images (augmentation, normalization)
        all_unique_images = np.stack(frames, axis=0)
        all_unique_images = preprocess_image(
            all_unique_images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        # Expand by repeats
        lengths = torch.as_tensor(repeats, dtype=torch.long, device=all_unique_images.device)
        all_images = torch.repeat_interleave(all_unique_images, lengths, dim=1)

        # Get action chunk
        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # Get value return
        if self.return_value_function_returns:
            value_function_return = episode_data["returns"][future_frame_idx]
        else:
            value_function_return = float("-100")

        # Next action chunk
        next_relative_step_idx = min(relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1)
        next_action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=next_relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )
        next_future_frame_idx = min(next_relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1)
        if self.return_value_function_returns:
            next_value_function_return = episode_data["returns"][next_future_frame_idx]
        else:
            next_value_function_return = float("-100")

        # T5 text embedding
        command = episode_data["command"]
        if command in self.t5_text_embeddings:
            t5_emb = torch.squeeze(self.t5_text_embeddings[command])
        else:
            t5_emb = torch.zeros(512, 1024)  # T5 embedding dim

        # Rollout masks
        rollout_data_mask = 0 if sample_type == "demo" else 1
        rollout_data_success_mask = 1 if sample_type == "success_rollout" else 0

        # World model vs value function sample
        # For demos: both False (action is denoised/predicted, not conditioned)
        # For rollouts: randomly world_model or value_function
        is_world_model_sample = False
        is_value_function_sample = False
        if sample_type != "demo" and self.return_value_function_returns:
            if np.random.rand() < 0.5:
                is_world_model_sample = True
                is_value_function_sample = False
            else:
                is_world_model_sample = False
                is_value_function_sample = True

        return {
            "video": all_images,
            "command": command,
            "actions": action_chunk,
            "t5_text_embeddings": t5_emb,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio,
            "future_proprio": future_proprio,
            "__key__": idx,
            "value_function_return": value_function_return,
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": next_value_function_return,
            "rollout_data_mask": rollout_data_mask,
            "rollout_data_success_mask": rollout_data_success_mask,
            "world_model_sample_mask": 1 if is_world_model_sample else 0,
            "value_function_sample_mask": 1 if is_value_function_sample else 0,
            "global_rollout_idx": -1,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": value_latent_idx,
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_wrist_image2_latent_idx": current_wrist_image2_latent_idx,
            "current_image_latent_idx": current_image_latent_idx,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx,
            "future_wrist_image2_latent_idx": future_wrist_image2_latent_idx,
            "future_image_latent_idx": future_image_latent_idx,
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/workspace/gps_training_data"

    dataset = AigenGPSDrivingDataset(
        data_dir=data_dir,
        chunk_size=50,
        final_image_size=224,
        use_proprio=True,
        normalize_actions=True,
        normalize_proprio=True,
        return_value_function_returns=True,
        debug=True,
    )
    print(f"\nDataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Video shape: {sample['video'].shape}")
    print(f"Actions shape: {sample['actions'].shape}")
    print(f"Proprio shape: {sample['proprio'].shape}")
    print(f"Command: {sample['command']}")
    print(f"Action latent idx: {sample['action_latent_idx']}")
    print(f"Current image latent idx: {sample['current_image_latent_idx']}")
