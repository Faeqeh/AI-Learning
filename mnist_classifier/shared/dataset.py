from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()

full_train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

train_size = 50000
validation_size = 10000

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [train_size, validation_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)

# images, labels = next(iter(train_loader))
# print(images[0],labels[0])
# print("Train:", len(train_dataset))
# print("Validation:", len(validation_dataset))
# print("Test:", len(test_dataset))
# print("Images shape:", images.shape)
# print("Labels shape:", labels.shape)