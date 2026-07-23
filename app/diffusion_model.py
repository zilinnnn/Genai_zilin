import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timestep, dim=32, max_period=10000):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timestep.device)
        / half
    )
    args = timestep.float()[:, None] * frequencies[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SmallDenoiser(nn.Module):
    def __init__(self, time_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_embed = sinusoidal_embedding(t, dim=32)
        t_embed = self.time_mlp(t_embed)[:, :, None, None]
        h = F.silu(self.conv1(x) + t_embed)
        h = F.silu(self.conv2(h))
        return self.conv3(h)


class DiffusionImageGenerator:
    def __init__(self, device=None, weights_path="diffusion_model.pth"):
        self.device = device or torch.device("cpu")
        self.model = SmallDenoiser().to(self.device)
        try:
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            print("Diffusion model loaded.")
        except (FileNotFoundError, RuntimeError) as error:
            print(f"Diffusion model not loaded: {error}")
            print("Using untrained diffusion model.")
        self.model.eval()

    def generate(self, steps=30, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        image = torch.randn(1, 3, 64, 64, device=self.device)

        with torch.no_grad():
            for step in reversed(range(steps)):
                t = torch.tensor([step], device=self.device)
                predicted_noise = self.model(image, t)
                image = image - predicted_noise / max(steps, 1)
                image = image.clamp(-1.0, 1.0)

        image = (image + 1.0) / 2.0

        return {
            "image": image.cpu().squeeze(0).permute(1, 2, 0).tolist(),
            "image_shape": [64, 64, 3],
            "steps": steps,
        }
