# load_data_set.py
import os
import yaml
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import glob

class TrafficSignDataset(Dataset):
    def __init__(self, data_yaml, split='train', transform=None):
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            
        self.root_dir = os.path.dirname(data_yaml)
        self.split = split
        self.image_dir = data[split]
        self.nc = data['nc']
        self.class_names = data['names']
        self.transform = transform
        
        # Get list of image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
            self.image_files.extend(
                sorted(glob.glob(os.path.join(self.root_dir, self.image_dir, ext)))
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = img_path.replace('images', 'labels').replace(os.path.splitext(img_path)[1], '.txt')
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        boxes = []
        classes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([x_center, y_center, width, height])
                    classes.append(class_id)
        
        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        classes = torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: convert to tensor and normalize
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, {'boxes': boxes, 'labels': classes}

def test_dataset():
    # Load the dataset
    dataset = TrafficSignDataset('/home/salma/graduation_project/YOLO/yolov8/Self-Driving Cars.v1i.yolov8/data.yaml', split='train')
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Get a batch
    images, targets = next(iter(dataloader))
    
    print(f"Dataset contains {len(dataset)} images")
    print(f"Batch shape: {images.shape}")
    print(f"Number of classes: {dataset.nc}")
    print(f"Class names: {dataset.class_names}")
    
    # Visualize first sample
    img = images[0].permute(1, 2, 0).numpy()
    boxes = targets['boxes'][0]
    labels = targets['labels'][0]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    
    # Draw bounding boxes
    for box, label in zip(boxes, labels):
        x_center, y_center, width, height = box.numpy()
        x_min = int((x_center - width/2) * img.shape[1])
        y_min = int((y_center - height/2) * img.shape[0])
        x_max = int((x_center + width/2) * img.shape[1])
        y_max = int((y_center + height/2) * img.shape[0])
        
        plt.gca().add_patch(plt.Rectangle(
            (x_min, y_min), 
            x_max - x_min, 
            y_max - y_min,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        ))
        plt.text(x_min, y_min - 5, 
                f'{dataset.class_names[int(label)]}',
                color='red', fontsize=12, backgroundcolor='white')
    
    plt.axis('off')
    plt.title('Sample Image with Annotations')
    plt.show()

if __name__ == '__main__':
    test_dataset()
