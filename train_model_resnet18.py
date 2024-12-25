import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torch import nn

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Ensure bounding box columns are numeric
        bbox_columns = ["xmin", "ymin", "xmax", "ymax"]
        for col in bbox_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
        self.data = self.data.dropna(subset=bbox_columns).reset_index(drop=True)

        # Map classes to indices
        self.class_mapping = {"potato": 1, "apple": 2, "beans": 3, "banana": 4, "pasta": 5}

    def __len__(self):
        return len(self.data["image_path"].unique())

    def __getitem__(self, idx):
        # Group rows by image_path
        image_path = self.data["image_path"].unique()[idx]
        rows = self.data[self.data["image_path"] == image_path]

        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        image = Image.open(image_path).convert("RGB")

        # Extract bounding boxes and labels
        boxes = rows[["xmin", "ymin", "xmax", "ymax"]].values
        labels = rows["class"].map(self.class_mapping).values

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Create target dictionary
        target = {"boxes": boxes, "labels": labels}
        return image, target

# Define collate_fn for batching
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# Define ResNet-18 Backbone with FPN
def create_resnet18_backbone():
    # Load ResNet-18 
    resnet = resnet18(pretrained=True)
    
    # Define return layers as per original ResNet-18 layer names
    return_layers = {
        "layer1": "0",
        "layer2": "1", 
        "layer3": "2", 
        "layer4": "3"
    }

    # Define FPN input channels 
    in_channels_list = [64, 128, 256, 512]  # Output channels from ResNet-18 layers
    out_channels = 256  # Desired FPN output channels

    # Create BackboneWithFPN
    backbone = BackboneWithFPN(
        resnet,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool()
    )
    
    return backbone

# Define the Faster R-CNN model
def create_model(num_classes):
    backbone = create_resnet18_backbone()
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

# Train function
def train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0

    for images, targets in tqdm(data_loader):
        # Move data to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass stage
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass stage
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)

# The main function
def main():
    # Paths
    csv_file = r"C:\Users\rdhop\Documents\DE4\SIOT\final_annotations.csv"  #Replace this with File path to csv with bounding box and image data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset(csv_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Model, optimiser, and scheduler
    model = create_model(num_classes=6)  # 5 classes + background = 6 classes
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = train(model, data_loader, optimiser, device)
        print(f"Loss: {epoch_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), r"C:\Users\rdhop\Documents\DE4\SIOT\final_model_resnet18.pth")
    print("Model saved!")

if __name__ == "__main__":
    main()