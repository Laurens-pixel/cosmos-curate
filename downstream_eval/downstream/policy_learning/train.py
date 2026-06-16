# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train a language-conditioned diffusion policy on a curated ``LeRobotDataset``.

Uses LeRobot's ``DiffusionPolicy`` with a standard torch training loop over the dataset
produced by :mod:`lerobot_adapter`. This is the *real* training path; it requires
``lerobot`` + ``torch`` + a GPU and is therefore gated behind lazy imports. The synthetic
smoke path (``synthetic.py``) does not need this.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """Configuration for diffusion-policy training."""

    dataset_repo_id: str
    dataset_root: str | Path
    output_dir: str | Path
    steps: int = 20_000
    batch_size: int = 64
    lr: float = 1e-4
    device: str = "cuda"
    log_every: int = 200


def train_diffusion_policy(config: TrainConfig) -> Path:
    """Train a diffusion policy and return the path to the saved checkpoint.

    Raises:
        RuntimeError: If lerobot/torch are unavailable.

    """
    try:
        import torch
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    except ImportError as exc:  # pragma: no cover - needs the optional stack
        msg = "train_diffusion_policy requires 'lerobot' and 'torch' (+ GPU)."
        raise RuntimeError(msg) from exc

    dataset = LeRobotDataset(config.dataset_repo_id, root=Path(config.dataset_root))
    policy_cfg = DiffusionConfig()
    policy = DiffusionPolicy(policy_cfg, dataset_stats=dataset.meta.stats).to(config.device)
    policy.train()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)

    step = 0
    done = False
    while not done:
        for batch in loader:
            batch = {k: (v.to(config.device) if hasattr(v, "to") else v) for k, v in batch.items()}
            loss = policy.forward(batch)["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % config.log_every == 0:
                print(f"[train] step={step} loss={float(loss):.4f}")  # noqa: T201
            step += 1
            if step >= config.steps:
                done = True
                break

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    return out
