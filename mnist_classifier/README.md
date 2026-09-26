# MNIST Digit Classification with PyTorch

A handwritten digit classification project implemented with **PyTorch**, featuring two different neural network approaches:

* **MLP (Multi-Layer Perceptron)**
* **CNN (Convolutional Neural Network)**

The project demonstrates the complete machine learning workflow from data preparation and model training to evaluation and error analysis.

## Models

### MLP

A fully connected neural network for classifying MNIST handwritten digits.

**Architecture:**

```text
784 → 128 → 64 → 10
```

[View MLP Project](mlp/README.md)

### CNN

A convolutional neural network designed to learn spatial features from handwritten digit images.

**Architecture:**

```text
Conv2D → ReLU → MaxPool
        ↓
Conv2D → ReLU → MaxPool
        ↓
Flatten → Linear → ReLU → Linear
```

[View CNN Project](cnn/README.md)

## MLP vs CNN

The project compares two neural network architectures for MNIST digit classification.

| Metric                   |                 MLP |                                                                        CNN |
| ------------------------ | ------------------: | -------------------------------------------------------------------------: |
| Architecture             | 784 → 128 → 64 → 10 | Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Linear → ReLU → Linear |
| Trainable Parameters     |             109,386 |                                                                    421,642 |
| Training Time (5 epochs) |       55.01 seconds |                                                             266.69 seconds |
| Validation Accuracy      |              98.33% |                                                                     99.45% |
| Validation Precision     |              98.34% |                                                                     99.46% |
| Validation Recall        |              98.29% |                                                                     99.45% |
| Validation F1 Score      |              98.31% |                                                                     99.45% |
| Test Accuracy            |              97.36% |                                                                     98.93% |
| Test Precision           |              97.36% |                                                                     98.94% |
| Test Recall              |              97.32% |                                                                     98.92% |
| Test F1 Score            |              97.33% |                                                                     98.92% |

### Architecture Comparison

**MLP**

```text
Input: 28 × 28
      ↓
Flatten: 784
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
```

**CNN**

```text
Input: 1 × 28 × 28
      ↓
Conv2D: 1 → 32
      ↓
ReLU → MaxPool
      ↓
Conv2D: 32 → 64
      ↓
ReLU → MaxPool
      ↓
Flatten: 64 × 7 × 7
      ↓
Linear: 3136 → 128
      ↓
ReLU
      ↓
Linear: 128 → 10
```

The CNN contains more trainable parameters and requires more training time in this experiment. It preserves the spatial structure of the input image through convolutional layers, while the MLP first flattens the image into a one-dimensional vector.

In the recorded experiments, the CNN achieved higher validation and test metrics than the MLP on the MNIST dataset.

## Dataset

The project uses the **MNIST handwritten digit dataset**.

| Dataset    | Samples |
| ---------- | ------: |
| Training   |  50,000 |
| Validation |  10,000 |
| Test       |  10,000 |

Each image is a grayscale image with a resolution of **28 × 28 pixels**.

## Technologies

* Python
* PyTorch
* Torchvision
* Scikit-learn
* Matplotlib

## Project Structure

```text
mnist-classifier/
│
├── mlp/
│   ├── README.md
│   ├── src/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── models/
│
├── cnn/
│   ├── README.md
│   ├── src/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── models/
│
├── shared/
│   └── dataset.py
│
├── data/
├── README.md
├── requirements.txt
└── .gitignore
```

## Machine Learning Workflow

```text
MNIST Dataset
      ↓
Data Preparation
      ↓
Train / Validation / Test Split
      ↓
Model Training
      ↓
Validation
      ↓
Test Evaluation
      ↓
Confusion Matrix
      ↓
Prediction & Error Analysis
```

## Project Goals

This project was created to practice and demonstrate:

* Working with image datasets
* Building neural networks with PyTorch
* Training models using backpropagation
* Using optimizers and loss functions
* Model validation and evaluation
* Classification metrics
* Confusion matrix analysis
* Prediction and error analysis
* Comparing different neural network architectures

## Results

The two models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

Detailed results, training curves, confusion matrices, predictions, and misclassified examples are available in the individual project READMEs.

## Future Improvements

* Hyperparameter tuning
* Data augmentation
* Experimenting with different CNN architectures
* Comparing additional optimization strategies
* Improving model generalization
