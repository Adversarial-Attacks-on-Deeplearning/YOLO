import torch
from load_data_set import TrafficSignDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

# Define image transformations
IMAGE_SIZE = 640  # Match YOLOv8's input size
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Initialize dataset and dataloader
def get_dataloader(data_yaml_path, split='train', batch_size=8):
    dataset = TrafficSignDataset(
        data_yaml=data_yaml_path,
        split=split,
        transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)



#helper functions
def preprocess(x):
    """Prepare images for EfficientNet discriminator"""
    # Scale from [0,1] to [-1,1] if using Tanh generator
    x = (x * 2) - 1
    return TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def denormalize(x):
    """Convert generator output to YOLO-compatible format"""
    # Scale from [-1,1] to [0,1]
    return (x + 1) / 2



if __name__ == '__main__':
    # Example usage
    data_yaml_path = "/home/salma/graduation_project/YOLO/yolov8/Self-Driving Cars.v1i.yolov8/data.yaml"
    train_loader = get_dataloader(data_yaml_path, split='train', batch_size=16)
    val_loader = get_dataloader(data_yaml_path, split='val', batch_size=16)
    test_loader = get_dataloader(data_yaml_path, split='test', batch_size=16)
    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images, Test: {len(test_loader.dataset)} images")