from dataclasses import dataclass, field

@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02

@dataclass
class TrainConfig:
    task_name: str = 'default'
    batch_size: int = 32  # Adjusted for 64x64 RGB images
    num_epochs: int = 40
    num_samples: int = 100
    num_grid_rows: int = 10
    lr: float = 0.0001
    ckpt_name: str = 'ddpm_ckpt.pth'

@dataclass
class UnetConfig:
    im_channels: int = 3  # Changed from 1 to 3 for RGB images (ALOT dataset)
    im_size: int = 64  # Changed from 28 to 64 for ALOT dataset (256 too large for attention)
    # doing field cos lists r mutable and i DONT want them shared
    down_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    mid_channels: list = field(default_factory=lambda: [256, 256, 128])
    down_sample: list = field(default_factory=lambda: [True, True, False])
    time_emb_dim: int = 128
    #num_down_layers: int = 2
    #num_mid_layers: int = 2
    #num_up_layers: int = 2
    num_heads: int = 4