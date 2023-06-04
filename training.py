# training.py

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Import shared dependencies
from object_detection import YOLOv5
from preprocessing import NormalizeMouseCoordinates
from data_collection import CollectMouseMovementData

# Define the training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss

# Define the main function
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the training data
    data = CollectMouseMovementData.load_data("mouse_movement_data.csv")
    data = NormalizeMouseCoordinates.normalize(data)

    # Define the YOLOv5 model
    model = YOLOv5().to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Define the data loader
    train_loader = DataLoader(data, batch_size=32, shuffle=True)

    # Train the model
    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "yolov5_trained.pt")

if __name__ == "__main__":
    main()