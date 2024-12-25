import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Custom Dataset
class CustomDataset:
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
        image_path = self.data["image_path"].unique()[idx]
        rows = self.data[self.data["image_path"] == image_path]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        image = Image.open(image_path).convert("RGB")

        boxes = rows[["xmin", "ymin", "xmax", "ymax"]].values
        labels = rows["class"].map(self.class_mapping).values

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes, "labels": labels}
        return image, target

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def create_resnet18_backbone():
    resnet = resnet18(pretrained=True)
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [64, 128, 256, 512]
    out_channels = 256

    backbone = BackboneWithFPN(
        resnet,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels
    )
    return backbone

def create_model(num_classes):
    backbone = create_resnet18_backbone()
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

def calculate_iou(boxA, boxB):
    # Calculate the intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate the union
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    # Compute IoU
    return interArea / unionArea if unionArea > 0 else 0

def evaluate_iou(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    class_iou_scores = {i: [] for i in range(1, 6)}  # Assuming 5 classes

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                pred_boxes = pred["boxes"].cpu().numpy()
                pred_labels = pred["labels"].cpu().numpy()
                target_boxes = target["boxes"].numpy()
                target_labels = target["labels"].numpy()

                for t_box, t_label in zip(target_boxes, target_labels):
                    ious = [
                        calculate_iou(t_box, p_box)
                        for p_box, p_label in zip(pred_boxes, pred_labels)
                        if p_label == t_label
                    ]
                    if ious:
                        max_iou = max(ious)
                        class_iou_scores[t_label].append(max_iou)

    # Calculate average IoU for each class
    avg_iou_per_class = {
        cls: np.mean(scores) if scores else 0 for cls, scores in class_iou_scores.items()
    }
    return avg_iou_per_class

def plot_iou_scores(avg_iou_per_class, class_mapping):
    class_names = [name for name, idx in class_mapping.items()]
    avg_ious = [avg_iou_per_class[idx] for name, idx in class_mapping.items()]

    plt.figure(figsize=(10, 6))
    plt.bar(class_names, avg_ious, color="skyblue")
    plt.xlabel("Class")
    plt.ylabel("Average IoU")
    plt.title("Average IoU per Class")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def main():
    # Paths
    model_path = r"C:\Users\rdhop\Documents\DE4\SIOT\final_model_resnet18.pth" #final model weights path
    test_csv = r"C:\Users\rdhop\Documents\DE4\SIOT\test_set.csv" # path to test set annotations csv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test dataset and dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = CustomDataset(test_csv, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Load model
    model = create_model(num_classes=6)  # 5 classes + background
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Evaluate IoU
    print("Calculating IoU...")
    avg_iou_per_class = evaluate_iou(model, test_loader, device)

    # Print and plot results
    print("\nAverage IoU per Class:")
    for cls, iou in avg_iou_per_class.items():
        class_name = {v: k for k, v in test_dataset.class_mapping.items()}[cls]
        print(f"{class_name}: {iou:.2f}")

    print("\nPlotting IoU scores...")
    plot_iou_scores(avg_iou_per_class, test_dataset.class_mapping)

if __name__ == "__main__":
    main()
