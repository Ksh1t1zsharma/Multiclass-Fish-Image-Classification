import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

DEFAULT_IMG_SIZE = 224

def get_transforms(img_size=DEFAULT_IMG_SIZE, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transforms, val_transforms

def make_dataloaders(data_dir, batch_size=32, img_size=DEFAULT_IMG_SIZE, num_workers=4, val_split=0.2, test_split=0.1):
    """
    Expects ImageFolder-style layout:
    data_dir/
      train/ (or all images to be split)
        class_x/
        class_y/
    If separate val/test folders exist, set val_split/test_split to 0 and pass explicit paths.
    """
    train_t, val_t = get_transforms(img_size)
    dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_t)

    # Create train/val split if val folder not provided
    total = len(dataset)
    val_len = int(total * val_split)
    train_len = total - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    # For val dataset, apply validation transforms (replace transform)
    val_ds.dataset.transform = val_t

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Try to load test set if present
    test_loader = None
    test_folder = os.path.join(data_dir, "test")
    if os.path.isdir(test_folder):
        test_ds = datasets.ImageFolder(test_folder, transform=val_t)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = dataset.classes
    return train_loader, val_loader, test_loader, class_names
