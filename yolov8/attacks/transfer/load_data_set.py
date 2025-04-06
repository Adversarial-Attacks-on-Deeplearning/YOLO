import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class YOLODataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Custom dataset for loading images.

        Args:
            image_dir (str): Path to the image directory.
            transform (callable, optional): Transformations to apply.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def get_data_loaders(train_dir, valid_dir, test_dir, batch_size=32, image_size=256):
    """
    Creates PyTorch DataLoaders for GAN training.

    Args:
        train_dir (str): Directory for training images.
        valid_dir (str): Directory for validation images.
        test_dir (str): Directory for test images.
        batch_size (int): Batch size for training.
        image_size (int): Resizing target for images.

    Returns:
        dict: Dataloaders for train, validation, and test sets.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize images
        transforms.ToTensor(),  # Convert to tensor
        
    ])

    train_dataset = YOLODataset(train_dir, transform=transform)
    valid_dataset = YOLODataset(valid_dir, transform=transform)
    test_dataset = YOLODataset(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return {"train": train_loader, "valid": valid_loader, "test": test_loader}

if __name__ == "__main__":
    # Define your directories as arguments
    train_dir = "/home/salma/graduation_project/YOLO/yolov8/Self-Driving Cars.v1i.yolov8/train/images"
    valid_dir = "/home/salma/graduation_project/YOLO/yolov8/Self-Driving Cars.v1i.yolov8/valid/images"
    test_dir = "/home/salma/graduation_project/YOLO/yolov8/Self-Driving Cars.v1i.yolov8/test/images"

    # Load dataset
    dataloaders = get_data_loaders(train_dir, valid_dir, test_dir, batch_size=16, image_size=256)

    # Get training batch
    for batch in dataloaders["train"]:
        print(batch.shape)  # Expected: (batch_size, 3, 256, 256)
        break
