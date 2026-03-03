# Aigen GPS Driving experiment configs for Cosmos Policy
#
# Single-camera, 2-DOF navigation robot. Much simpler than ALOHA.
# state_t=7: blank, proprio, primary, action, future_proprio, future_primary, value
# chunk_duration=25: 1 blank + 6*4 latent frames

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_policy.datasets.aigen_gps_dataset import AigenGPSDrivingDataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE

BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")

# Dataset config
aigen_gps_driving_dataset = L(AigenGPSDrivingDataset)(
    data_dir="/data/cosmos_training",
    t5_text_embeddings_path="/data/cosmos_training/t5_embeddings.pkl",
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer
    treat_demos_as_success_rollouts=True,
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=False,
    gamma=0.99,
)

# Training config — single GPU (RTX 3090, 24GB)
cosmos_predict2_2b_480p_aigen_gps_driving = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-102-Size-2B-Res-480-Fps-16-Note-HQ_V5_from_26",
            {"override /data_train": "mock"},
            {"override /model": "policy_fsdp"},
            {"override /tokenizer": "policy_wan2pt1_tokenizer"},
            {
                "override /callbacks": [
                    "basic",
                    "long",
                    "cluster_speed",
                    "wandb",
                    "wandb_callback_actions",
                ]
            },
            "_self_",
        ],
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=50000,
                    save_s3=False,
                    use_negative_prompt=False,
                    guidance=[0],
                    num_sampling_step=9,
                ),
            ),
            run_validation=False,
            logging_iter=5,
            max_iter=100000,  # Less than ALOHA since simpler task
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 15K steps, then constant
            cycle_lengths=[15000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.0,  # Don't drop text conditioning
                    ),
                ),
                # LoRA finetuning (freeze base, train adapters)
                use_lora=True,
                lora_rank=32,
                lora_alpha=32,
                # Simpler than ALOHA: no wrist cameras, no value function
                # blank, proprio, primary, action, future_proprio, future_primary
                state_t=6,
                min_num_conditional_frames=3,  # 1 blank + 2 conditioning (proprio, primary)
                max_num_conditional_frames=3,
                sigma_conditional=0.0,
                conditioning_strategy="frame_replace",
                denoise_replace_gt_frames=True,
                tokenizer=dict(
                    chunk_duration=21,  # 1 blank + 5*4 = 21 (no value)
                ),
                ema=dict(
                    enabled=False,
                ),
                input_data_key="video",
                sde=L(HybridEDMSDE)(
                    hybrid_sigma_distribution=True,
                    p_mean=1.3862943611198906,
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                    uniform_lower=1.0,
                    uniform_upper=85.0,
                ),
                adjust_video_noise=True,
                resize_online=True,
                resolution="224",
                high_sigma_strategy="none",
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            # NOTE: checkpoint will be resolved at training time, not import time
            # Use: get_checkpoint_path("hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt")
            # For now, set to empty string and override via CLI
            load_path=get_checkpoint_path("hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"),
            load_training_state=False,
            strict_resume=False,
            save_iter=1000,
            load_ema_to_reg=True,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,  # Fewer workers for single GPU
            persistent_workers=True,
            pin_memory=True,
            dataset=aigen_gps_driving_dataset,
            sampler=L(DistributedSampler)(
                dataset=aigen_gps_driving_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=1,  # Start with 1 for 24GB VRAM, increase if fits
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_aigen_gps_driving",
        ),
        upload_reproducible_setup=False,
    )
)

# Inference version
cosmos_predict2_2b_480p_aigen_gps_driving__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aigen_gps_driving",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_aigen_gps_driving__inference_only",
        ),
    )
)


def register_aigen_configs():
    cs = ConfigStore.instance()
    for _item in [
        cosmos_predict2_2b_480p_aigen_gps_driving,
        cosmos_predict2_2b_480p_aigen_gps_driving__inference_only,
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering Aigen experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
