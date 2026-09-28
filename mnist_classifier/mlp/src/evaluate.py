import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

from mlp.src.model import MNISTModel
from shared.dataset import validation_loader, test_loader


def evaluate(model, data_loader):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = sum(
        prediction == label
        for prediction, label in zip(all_predictions, all_labels)
    ) / len(all_labels)

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions,
    )


model = MNISTModel()

model.load_state_dict(
    torch.load("models/mnist_model.pth")
)

validation_metrics = evaluate(
    model,
    validation_loader,
)

test_metrics = evaluate(
    model,
    test_loader,
)


print("\nValidation:")
print(f"Accuracy:  {validation_metrics[0]:.4f}")
print(f"Precision: {validation_metrics[1]:.4f}")
print(f"Recall:    {validation_metrics[2]:.4f}")
print(f"F1 Score:  {validation_metrics[3]:.4f}")


print("\nTest:")
print(f"Accuracy:  {test_metrics[0]:.4f}")
print(f"Precision: {test_metrics[1]:.4f}")
print(f"Recall:    {test_metrics[2]:.4f}")
print(f"F1 Score:  {test_metrics[3]:.4f}")


test_labels = test_metrics[4]
test_predictions = test_metrics[5]

cm = confusion_matrix(
    test_labels,
    test_predictions,
)

# print("\nConfusion Matrix:")
# print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10),
)

display.plot()

plt.title("MNIST Test Confusion Matrix")
plt.savefig(
    "models/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
