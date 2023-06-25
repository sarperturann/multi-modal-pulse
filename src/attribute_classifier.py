import argparse
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def train_attribute_classifier(data_dir, output_file):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the ResNet model
    model = models.resnet18(pretrained=True)
    num_classes = 2  # Number of output classes (e.g., smiling or not smiling)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Define data transformations
    transform = ToTensor()

    # Load and prepare the dataset
    train_dataset = ImageFolder(data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs, latent_vectors = model(
                images, return_latent=True)  # Obtain latent vectors

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    # Save the latent vectors to a file
    torch.save(latent_vectors, output_file)


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Train attribute classifier and save latent vectors.')

    # Add command line arguments
    parser.add_argument('--data_dir', type=str,
                        help='Path to the dataset directory')
    parser.add_argument('--output_file', type=str,
                        help='Path to save the output latent vectors')

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function
    train_attribute_classifier(args.data_dir, args.output_file)
