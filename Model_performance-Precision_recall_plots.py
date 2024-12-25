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
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from collections import defaultdict


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

# Model creation (needed to load the saved model)
def create_resnet18_backbone():
    resnet = resnet18(pretrained=True)
    return_layers = {
        "layer1": "0",
        "layer2": "1", 
        "layer3": "2", 
        "layer4": "3"
    }
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

def create_model(num_classes):
    backbone = create_resnet18_backbone()
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

def evaluate_model(model, data_loader, device, confidence_threshold=0.5):
    model.eval()
    all_predictions = defaultdict(list)
    all_targets = defaultdict(list)
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                pred_boxes = pred['boxes'].cpu()
                pred_scores = pred['scores'].cpu()
                pred_labels = pred['labels'].cpu()
                target_boxes = target['boxes']
                target_labels = target['labels']
                
                # Filter predictions by confidence threshold
                mask = pred_scores > confidence_threshold
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]
                
                # Get unique classes from both predictions and targets
                unique_classes = torch.unique(torch.cat([target_labels, pred_labels]))
                
                for class_idx in unique_classes:
                    class_idx = class_idx.item()
                    
                    # Count target instances for this class
                    target_count = (target_labels == class_idx).sum().item()
                    
                    # Count prediction instances for this class
                    pred_mask = pred_labels == class_idx
                    pred_count = pred_mask.sum().item()
                    
                    # Add scores for predictions of this class
                    if pred_count > 0:
                        all_predictions[class_idx].extend(pred_scores[pred_mask].numpy())
                        # Add 1s for true positives (up to the number of actual targets)
                        all_targets[class_idx].extend([1] * min(target_count, pred_count))
                        # Add 0s for false positives
                        if pred_count > target_count:
                            all_targets[class_idx].extend([0] * (pred_count - target_count))
                    
                    # If we have targets but no predictions, add zeros
                    if target_count > pred_count:
                        all_predictions[class_idx].extend([0] * (target_count - pred_count))
                        all_targets[class_idx].extend([1] * (target_count - pred_count))

    # Convert lists to numpy arrays and ensure equal lengths
    for class_idx in all_predictions:
        all_predictions[class_idx] = np.array(all_predictions[class_idx])
        all_targets[class_idx] = np.array(all_targets[class_idx])
        
        # Double-check lengths
        assert len(all_predictions[class_idx]) == len(all_targets[class_idx]), \
            f"Mismatch in lengths for class {class_idx}: {len(all_predictions[class_idx])} vs {len(all_targets[class_idx])}"
    
    return all_predictions, all_targets

def plot_precision_recall_curves(all_predictions, all_targets, class_mapping):
    plt.figure(figsize=(10, 8))
    rev_class_mapping = {v: k for k, v in class_mapping.items()}
    
    for class_idx in all_predictions.keys():
        if len(all_predictions[class_idx]) == 0:
            continue
            
        try:
            precision, recall, _ = precision_recall_curve(
                all_targets[class_idx], 
                all_predictions[class_idx]
            )
            ap = average_precision_score(
                all_targets[class_idx], 
                all_predictions[class_idx]
            )
            
            plt.plot(recall, precision, 
                    label=f'{rev_class_mapping[class_idx]} (AP={ap:.2f})')
        except Exception as e:
            print(f"Error calculating metrics for class {rev_class_mapping[class_idx]}: {e}")
            print(f"Predictions shape: {all_predictions[class_idx].shape}")
            print(f"Targets shape: {all_targets[class_idx].shape}")
            continue
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves by Class')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    
    model_path = r"C:\Users\rdhop\Documents\DE4\SIOT\final_model_resnet18.pth" # Path to custom weights .pth file
    test_csv = r"C:\Users\rdhop\Documents\DE4\SIOT\test_set.csv" # path to test set annotations csv file
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader for test data
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = CustomDataset(test_csv, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Load model
    model = create_model(num_classes=6)  # 5 classes + background
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Evaluate model
    print("Evaluating model on test set...")
    all_predictions, all_targets = evaluate_model(model, test_loader, device)

    # Calculate and display metrics
    print("\nPer-class Average Precision:")
    mAP = 0
    valid_classes = 0
    
    for class_idx in all_predictions.keys():
        if len(all_predictions[class_idx]) > 0:
            try:
                ap = average_precision_score(
                    all_targets[class_idx], 
                    all_predictions[class_idx]
                )
                mAP += ap
                valid_classes += 1
                class_name = {v: k for k, v in test_dataset.class_mapping.items()}[class_idx]
                print(f"{class_name}: {ap:.3f}")
                print(f"Number of samples for {class_name}: {len(all_predictions[class_idx])}")
            except Exception as e:
                print(f"Error calculating AP for class {class_idx}: {e}")

    if valid_classes > 0:
        mAP /= valid_classes
        print(f"\nMean Average Precision (mAP): {mAP:.3f}")

    # Plot precision-recall curves
    print("\nGenerating precision-recall curves...")
    plot_precision_recall_curves(all_predictions, all_targets, test_dataset.class_mapping)

if __name__ == "__main__":
    main()