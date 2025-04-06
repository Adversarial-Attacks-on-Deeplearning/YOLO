import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import os


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    

class GENERATOR(nn.Module):
    def __init__(
        self, 
        in_channels=3,  # Changed from 1 to 3 (RGB input)
        out_channels=3, # Changed from 1 to 3 (RGB perturbation/output)
        features=[64, 128, 256, 512]
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups_transpose = nn.ModuleList()
        self.ups_conv = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path (unchanged except for in_channels)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature  # Propagate channel size

        # Upsampling path (unchanged)
        for feature in reversed(features):
            self.ups_transpose.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups_conv.append(DoubleConv(feature*2, feature))

        # Bottleneck (unchanged)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Final layer: Critical change for UEA
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Tanh()  # Added Tanh to bound output to [-1, 1] (GAN stability)
        )

    def forward(self, x):
        skip_connections = []

        # Downsampling (unchanged)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck (unchanged)
        x = self.bottleneck(x)

        # Upsampling (unchanged except for shape checks)
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.ups_transpose)):
            x = self.ups_transpose[idx](x)
            skip_connection = skip_connections[idx]

            # Handle shape mismatches (e.g., odd input dimensions)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])  # Resize for alignment

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups_conv[idx](concat_skip)

        return self.final_conv(x)  # Output: Adversarial perturbation in [-1, 1]
    

if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))  # Batch x 3 (RGB) x H x W
    model = GENERATOR(in_channels=3, out_channels=3)
    preds = model(x)
    print(f"Input shape: {x.shape}")    # [1, 3, 256, 256]
    print(f"Output shape: {preds.shape}")  # [1, 3, 256, 256]
    print(f"Output range: [{preds.min():.2f}, {preds.max():.2f}]")  # Should be ~[-1, 1]
    if x.shape == preds.shape:
        print("Output shape matches input shape!")
    else:
        print("Output shape does not match input shape!")