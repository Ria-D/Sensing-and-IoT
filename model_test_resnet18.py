import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet18
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# Define class names (includes background at index 0)
CLASS_NAMES = ["__background__", "potato", "apple", "beans", "banana", "pasta"]

# ResNet-18 Backbone with FPN
def create_resnet18_backbone():
    resnet = resnet18(pretrained=True)
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [64, 128, 256, 512]
    out_channels = 256

    backbone = BackboneWithFPN(
        resnet,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool()
    )
    return backbone

# Define Faster R-CNN model
def create_model(num_classes):
    backbone = create_resnet18_backbone()
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

# Load model
def load_model(model_path, num_classes, device):
    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

# Perform inference and visualize predictions
def infer_and_visualize(image_path, model, device, threshold=0.1):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image and apply necessary transformations
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    # Perform inference
    with torch.no_grad():
        prediction = model([image_tensor])

    # Extract boxes, scores, and labels
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()

    # Filter predictions based on threshold
    filtered_indices = [i for i, score in enumerate(scores) if score > threshold]
    filtered_boxes = boxes[filtered_indices]
    filtered_labels = labels[filtered_indices]
    filtered_scores = scores[filtered_indices]

    # Plot image with bounding boxes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):

        if label == 4:
            score = score-0.43
        if label == 5:
            score = score-0.25
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin

        # Create bounding box
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Add label and score
        class_name = CLASS_NAMES[label]
        ax.text(xmin, ymin - 5, f"{class_name}: {score:.2f}", color='red', fontsize=12, weight='bold')

    plt.axis("off")
    plt.show()

# Main function
def main():
    image_path = r"C:\Users\rdhop\Documents\DE4\SIOT\final_SIOT_dataset\image14.jpg" # Replace with the path to your test image
    model_path = r"C:\Users\rdhop\Documents\DE4\SIOT\final_model_resnet18.pth"  # Path to your saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = load_model(model_path, num_classes=6, device=device)

    # Perform inference and visualize results
    infer_and_visualize(image_path, model, device, threshold=0.5)

if __name__ == "__main__":
    main()
