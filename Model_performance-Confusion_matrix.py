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
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# IoU computation
def compute_iou(box1, box2):
    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Match predictions to ground truths
def match_predictions_to_ground_truth(pred_boxes, pred_labels, true_boxes, true_labels, iou_threshold=0.5):
    matched_preds = []
    matched_gts = []
    used_true_indices = set()

    for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        best_iou = 0
        best_true_idx = -1

        for true_idx, (true_box, true_label) in enumerate(zip(true_boxes, true_labels)):
            if true_idx in used_true_indices:
                continue

            iou = compute_iou(pred_box, true_box)
            if iou > best_iou and iou >= iou_threshold and pred_label == true_label:
                best_iou = iou
                best_true_idx = true_idx

        if best_true_idx != -1:
            matched_preds.append(pred_label)
            matched_gts.append(true_labels[best_true_idx])
            used_true_indices.add(best_true_idx)
        else:
            matched_preds.append(pred_label)
            matched_gts.append(0)

    for true_idx, true_label in enumerate(true_labels):
        if true_idx not in used_true_indices:
            matched_preds.append(0)
            matched_gts.append(true_label)

    return matched_preds, matched_gts

# Evaluation loop
def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()

                true_boxes = target['boxes'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()

                matched_preds, matched_gts = match_predictions_to_ground_truth(
                    pred_boxes, pred_labels, true_boxes, true_labels, iou_threshold
                )

                all_predictions.extend(matched_preds)
                all_ground_truths.extend(matched_gts)

    return all_predictions, all_ground_truths

# Display metrics without the background class
def calculate_metrics(predictions, ground_truths, class_names):
    # Exclude the background class (label 0)
    filtered_predictions = [p for p, g in zip(predictions, ground_truths) if g != 0]
    filtered_ground_truths = [g for g in ground_truths if g != 0]

    conf_matrix = confusion_matrix(filtered_ground_truths, filtered_predictions, labels=range(1, len(class_names) + 1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Without Background)")
    plt.show()

    report = classification_report(filtered_ground_truths, filtered_predictions, target_names=class_names, zero_division=0)
    print("Classification Report (Without Background):\n", report)

# Main function
def main():
    csv_file = "test_set.csv"
    model_path = "final_model_resnet18.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = CustomDataset(csv_file, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = create_model(num_classes=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    predictions, ground_truths = evaluate_model(model, test_loader, device)

    class_names = ["potato", "apple", "beans", "banana", "pasta"]
    calculate_metrics(predictions, ground_truths, class_names)

if __name__ == "__main__":
    main()