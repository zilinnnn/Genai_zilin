import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from app.gan_model import Generator, Discriminator


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    z_dim = 100
    learning_rate = 0.0002
    epochs = 5

    generator = Generator(z_dim=z_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()

    optimizer_generator = optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999)
    )

    optimizer_discriminator = optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999)
    )

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        for batch_index, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            optimizer_discriminator.zero_grad()

            real_predictions = discriminator(real_images)
            real_loss = criterion(real_predictions, real_labels)

            noise = torch.randn(batch_size, z_dim, device=device)
            fake_images = generator(noise)

            fake_predictions = discriminator(fake_images.detach())
            fake_loss = criterion(fake_predictions, fake_labels)

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            optimizer_discriminator.step()

            # Train Generator
            optimizer_generator.zero_grad()

            generator_predictions = discriminator(fake_images)

            generator_loss = criterion(
                generator_predictions,
                real_labels
            )

            generator_loss.backward()
            optimizer_generator.step()

            if batch_index % 200 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_index}/{len(dataloader)}] "
                    f"D Loss: {discriminator_loss.item():.4f} "
                    f"G Loss: {generator_loss.item():.4f}"
                )

        print(
            f"Epoch {epoch + 1}/{epochs} completed | "
            f"D Loss: {discriminator_loss.item():.4f} | "
            f"G Loss: {generator_loss.item():.4f}"
        )

    torch.save(
        generator.state_dict(),
        "gan_generator.pth"
    )

    torch.save(
        discriminator.state_dict(),
        "gan_discriminator.pth"
    )

    print("Training finished.")
    print("Saved gan_generator.pth")
    print("Saved gan_discriminator.pth")


if __name__ == "__main__":
    main()

