# MNIST Digit Classifier with PyTorch — MLP

## Overview

This project implements a handwritten digit classifier for the MNIST dataset using a **Multi-Layer Perceptron (MLP)** built with PyTorch.

The project demonstrates a complete machine learning workflow, including:

* Data preparation
* Training and validation
* Model evaluation
* Performance analysis
* Visualization of predictions and errors

## Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Scikit-learn
* Matplotlib

## Model Architecture

The MLP consists of three fully connected layers:

```text
Input Image
28 × 28 = 784
      ↓
Linear: 784 → 128
      ↓
ReLU
      ↓
Linear: 128 → 64
      ↓
ReLU
      ↓
Linear: 64 → 10
      ↓
Output
```

The output contains 10 values corresponding to the ten digit classes (`0–9`).

## Dataset

The project uses the **MNIST handwritten digit dataset**.

The dataset is divided into:

| Dataset    | Samples |
| ---------- | ------: |
| Training   |  50,000 |
| Validation |  10,000 |
| Test       |  10,000 |

Each image has a resolution of **28 × 28 pixels** and is converted to a tensor using `ToTensor()`.

## Training

The model is trained using:

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | Adam             |
| Learning Rate | 0.001            |
| Loss Function | CrossEntropyLoss |
| Batch Size    | 64               |
| Epochs        | 5                |

### Training Curves

#### Loss

The following plot shows the training and validation loss during training.

![Training and Validation Loss](models/loss_curve.png)

#### Validation Accuracy

![Validation Accuracy](models/accuracy_curve.png)

## Evaluation

The model is evaluated on both the validation and test datasets using:

* Accuracy
* Precision
* Recall
* F1 Score

### Results

| Metric    | Validation |   Test |
| --------- | ---------: | -----: |
| Accuracy  |     98.33% | 97.36% |
| Precision |     98.34% | 97.36% |
| Recall    |     98.29% | 97.32% |
| F1 Score  |     98.31% | 97.33% |

## Confusion Matrix

The confusion matrix shows the classification performance for each digit class on the MNIST test dataset.

![Confusion Matrix](models/confusion_matrix.png)

## Predictions

The following examples show correctly classified MNIST images together with their actual and predicted labels.

![Predictions](models/predictions.png)

## Misclassified Examples

The following examples show images that were incorrectly classified by the model.

![Misclassified Examples](models/misclassified.png)

## Project Structure

```text
mlp/
├── README.md
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
└── models/
    ├── mnist_model.pth
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── confusion_matrix.png
    ├── predictions.png
    └── misclassified.png
```

## How to Run

From the project root:

### Train the model

```bash
python -m mlp.src.train
```

### Evaluate the model

```bash
python -m mlp.src.evaluate
```

### Generate predictions and misclassified examples

```bash
python -m mlp.src.predict
```

## What This Project Demonstrates

This project demonstrates practical experience with:

* Building neural networks using PyTorch
* Preparing and loading image datasets
* Training models with backpropagation
* Using optimizers and loss functions
* Validating model performance
* Evaluating classification models
* Analyzing classification errors
* Visualizing machine learning results
