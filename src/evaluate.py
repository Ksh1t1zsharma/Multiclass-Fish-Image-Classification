import os
import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.data import get_transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models import get_model

def load_checkpoint(path, device):
    return torch.load(path, map_location=device)

def evaluate_checkpoint(checkpoint, model, loader, device, class_names, out_dir=None):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for images, t in loader:
            images = images.to(device)
            outputs = model(images)
            pred = outputs.argmax(1).cpu().numpy()
            preds.append(pred)
            targets.append(t.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    report = classification_report(targets, preds, target_names=class_names, digits=4)
    cm = confusion_matrix(targets, preds)
    print(report)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    return report, cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    val_transform = get_transforms(args.img_size)[1]
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    class_names = val_dataset.classes

    ckpt = load_checkpoint(args.checkpoint, device)
    model = get_model(args.model, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    evaluate_checkpoint(ckpt, model, val_loader, device, class_names, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
