import torch
import matplotlib.pyplot as plt

from cnn.src.model import MNISTCNN
from shared.dataset import test_dataset


def predict(model, image):
    model.eval()

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        prediction = output.argmax(dim=1).item()

    return prediction


model = MNISTCNN()

model.load_state_dict(
    torch.load("cnn/models/mnist_cnn.pth")
)


correct_predictions = []
misclassified = []


for index in range(len(test_dataset)):

    image, label = test_dataset[index]

    prediction = predict(model, image)

    if prediction == label:
        correct_predictions.append(
            (image, label, prediction)
        )
    else:
        misclassified.append(
            (image, label, prediction)
        )

    if (
        len(correct_predictions) == 10
        and len(misclassified) == 10
    ):
        break


# Correct Predictions

fig, axes = plt.subplots(
    2,
    5,
    figsize=(10, 5),
)

for ax, (image, label, prediction) in zip(
    axes.flat,
    correct_predictions,
):

    ax.imshow(
        image.squeeze(),
        cmap="gray",
    )

    ax.set_title(
        f"Actual: {label} | Predicted: {prediction}"
    )

    ax.axis("off")


plt.tight_layout()

plt.savefig(
    "cnn/models/predictions.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# Misclassified Predictions

fig, axes = plt.subplots(
    2,
    5,
    figsize=(10, 5),
)

for ax, (image, label, prediction) in zip(
    axes.flat,
    misclassified,
):

    ax.imshow(
        image.squeeze(),
        cmap="gray",
    )

    ax.set_title(
        f"Actual: {label} | Predicted: {prediction}"
    )

    ax.axis("off")


plt.tight_layout()

plt.savefig(
    "cnn/models/misclassified.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()