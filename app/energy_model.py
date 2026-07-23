import torch
import torch.nn as nn


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EnergyBasedImageGenerator:
    def __init__(self, device=None, weights_path="energy_model.pth"):
        self.device = device or torch.device("cpu")
        self.model = EnergyModel().to(self.device)
        try:
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            print("Energy model loaded.")
        except (FileNotFoundError, RuntimeError) as error:
            print(f"Energy model not loaded: {error}")
            print("Using untrained energy model.")
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def generate(self, steps=30, step_size=0.05, noise_scale=0.01, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        image = torch.rand(
            1,
            3,
            64,
            64,
            device=self.device,
            requires_grad=True,
        )
        energies = []

        for _ in range(steps):
            energy = self.model(image).sum()
            energies.append(float(energy.detach().cpu()))
            energy.backward()

            with torch.no_grad():
                image -= step_size * image.grad
                image += noise_scale * torch.randn_like(image)
                image.clamp_(0.0, 1.0)

            image.grad.zero_()

        return {
            "image": image.detach().cpu().squeeze(0).permute(1, 2, 0).tolist(),
            "image_shape": [64, 64, 3],
            "energies": energies,
            "steps": steps,
        }
