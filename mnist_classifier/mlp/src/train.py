import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt

from mlp.src.model import MNISTModel
from shared.dataset import train_loader, validation_loader


model = MNISTModel()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

epochs = 5

train_losses = []
validation_losses = []
validation_accuracies = []

start_time = time.time()

for epoch in range(epochs):

    # Training
    model.train()

    total_train_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )

    train_losses.append(average_train_loss)

    # Validation
    model.eval()

    total_validation_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_validation_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    average_validation_loss = (
        total_validation_loss / len(validation_loader)
    )

    validation_accuracy = correct / total

    validation_losses.append(average_validation_loss)
    validation_accuracies.append(validation_accuracy)

    print(
        f"Epoch [{epoch + 1}/{epochs}] | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Validation Loss: {average_validation_loss:.4f} | "
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")


# Save model
torch.save(
    model.state_dict(),
    "mlp/models/mnist_model.pth",
)

print("Model saved.")


# Loss Curve
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    train_losses,
    label="Train Loss",
)

plt.plot(
    range(1, epochs + 1),
    validation_losses,
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()

plt.savefig(
    "mlp/models/loss_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# Accuracy Curve
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    validation_accuracies,
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid()

plt.savefig(
    "mlp/models/accuracy_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()