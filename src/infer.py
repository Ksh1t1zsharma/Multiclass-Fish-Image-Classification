import os
import argparse
import csv
import torch
from PIL import Image
from torchvision import transforms
from src.models import get_model

DEFAULT_IMG_SIZE = 224

def load_image(path, img_size=DEFAULT_IMG_SIZE):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0)

def predict_image(model, path, device, img_size=DEFAULT_IMG_SIZE):
    img = load_image(path, img_size)
    img = img.to(device)
    model.eval()
    with torch.no_grad():
        out = model(img)
        pred = out.argmax(1).item()
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy().squeeze().tolist()
    return pred, probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--image", required=True, help="path to image or folder")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # We expect a checkpoints folder structure that doesn't include class names.
    ckpt = torch.load(args.checkpoint, map_location=device)
    # To get class names, infer from sibling folder if available or ask user to provide
    # For simplicity, require a class_names.txt in same folder as checkpoint (optional)
    ckpt_dir = os.path.dirname(args.checkpoint)
    class_file = os.path.join(ckpt_dir, "class_names.txt")
    if os.path.exists(class_file):
        with open(class_file, "r") as f:
            class_names = [l.strip() for l in f.readlines()]
    else:
        # fallback placeholder
        class_names = [str(i) for i in range(1000)]

    model = get_model(args.model, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    paths = []
    if os.path.isdir(args.image):
        for fn in sorted(os.listdir(args.image)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(args.image, fn))
    else:
        paths = [args.image]

    rows = []
    for p in paths:
        pred_idx, probs = predict_image(model, p, device, img_size=args.img_size)
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        rows.append({"image": p, "pred_idx": pred_idx, "pred_label": pred_label, "probs": probs})

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "pred_idx", "pred_label", "probs"])
            for r in rows:
                writer.writerow([r["image"], r["pred_idx"], r["pred_label"], r["probs"]])
    else:
        for r in rows:
            print(r)

if __name__ == "__main__":
    main()
