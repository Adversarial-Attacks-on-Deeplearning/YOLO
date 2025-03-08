from efficientnet_pytorch import EfficientNet
from torch import nn
import torch

class EfficientNetDiscriminator(nn.Module):
    def __init__(self, model_name='efficientnet-b0'):
        super().__init__()
        # Load pre-trained EfficientNet (or train from scratch)
        self.effnet = EfficientNet.from_pretrained(model_name)
        
        # Modify the final layer for binary classification
        in_features = self.effnet._fc.in_features
        self.effnet._fc = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()  # Output probability (0: fake, 1: real)
        )

    def forward(self, x):
        return self.effnet(x)
    
if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))  # Batch x 3 (RGB) x H x W
    model = EfficientNetDiscriminator()
    preds = model(x)
    print(preds)  # Output: Tensor([[0.5000]], grad_fn=<SigmoidBackward>)