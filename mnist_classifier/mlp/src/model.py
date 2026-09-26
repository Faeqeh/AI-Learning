import torch.nn as nn
import torch

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

if __name__ == "__main__":
    from shared.dataset import train_loader

    model = MNISTModel()
    images, labels = next(iter(train_loader))
    outputs = model(images)
    
    # print("Input shape:", images.shape)
    # print("Output shape:", outputs.shape)
    # print(labels)