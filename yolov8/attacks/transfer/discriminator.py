import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, image_height=256, image_width=256, n_channels=3):
        super(Discriminator, self).__init__()
        
        # Convolutional layers for 256x256 → 16x16
        self.conv1 = nn.Conv2d(n_channels, 64, 4, 2, 1)  # 256→128
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)         # 128→64
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)        # 64→32
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)        # 32→16
        
        # Dynamically calculate flattened size
        self.flattened_size = 512 * 16 * 16  # 512 channels × 16×16 spatial dim
        
        # Final linear layer
        self.fc = nn.Linear(self.flattened_size, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)  # [B, 64, 128, 128]
        x = F.leaky_relu(self.conv2(x), 0.2)   # [B, 128, 64, 64]
        x = F.leaky_relu(self.conv3(x), 0.2)   # [B, 256, 32, 32]
        x = F.leaky_relu(self.conv4(x), 0.2)   # [B, 512, 16, 16]
        x = x.view(x.size(0), -1)              # [B, 512*16*16]
        return self.fc(x)
    

if __name__ == "__main__":
    # Example usage
    # Initialize discriminator
    discriminator = Discriminator(image_height=256, image_width=256, n_channels=3)

    # Test with 256x256 input
    dummy_input = torch.randn(16, 3, 256, 256)  # Batch of 16 images
    output = discriminator(dummy_input)
    print(output.shape)  # Should be [16, 1]