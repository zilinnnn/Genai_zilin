import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from app.diffusion_model import SmallDenoiser


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
    diffusion_steps = 100

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmallDenoiser().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_index, (clean_images, _) in enumerate(loader):
            clean_images = clean_images.to(device)
            noise = torch.randn_like(clean_images)
            t = torch.randint(
                0,
                diffusion_steps,
                (clean_images.size(0),),
                device=device,
            )
            noise_amount = (t.float() / diffusion_steps).view(-1, 1, 1, 1)
            noisy_images = clean_images * (1 - noise_amount) + noise * noise_amount

            predicted_noise = model(noisy_images, t)
            loss = loss_fn(predicted_noise, noise)

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

    torch.save(model.state_dict(), "diffusion_model.pth")
    print("Saved diffusion_model.pth")


if __name__ == "__main__":
    main()
