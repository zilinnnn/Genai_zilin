import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from app.energy_model import EnergyModel


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    batch_size = 64
    epochs = 3
    learning_rate = 0.0005

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EnergyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_index, (real_images, _) in enumerate(loader):
            real_images = real_images.to(device)
            noise_images = torch.rand_like(real_images)

            real_energy = model(real_images)
            noise_energy = model(noise_images)

            logits = torch.cat([-real_energy, -noise_energy], dim=0)
            labels = torch.cat([
                torch.ones_like(real_energy),
                torch.zeros_like(noise_energy),
            ])

            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"Batch {batch_index}/{len(loader)} "
                    f"Loss: {loss.item():.4f}"
                )

        print(f"Epoch {epoch + 1} average loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "energy_model.pth")
    print("Saved energy_model.pth")


if __name__ == "__main__":
    main()
