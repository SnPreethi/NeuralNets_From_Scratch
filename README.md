# **BUILDING A DIGIT RECOGNIZER NEURAL NETWORK FROM SCRATCH USING NUMPY**


This repository contains an implementation of a simple feedforward neural network for recognizing handwritten digits using the MNIST dataset. The model is buit entirely from scratch using NumPy dataset, without using any deep learning frameworks like TensorFlow or PyTorch.

----

### **OVERVIEW**

This project implements a fully connected neural network (Feedforward Neural Network - FNN) with one hidden layer. The model is trained using gradient descent and utilizes ReLU and Softmax activation functions for classification.
The dataset consists of 28 x 28 grayscale images of handwritten digits (0-9), flattened into 784-dimensional feature vectors.

The model:

- Reads the dataset

- Preprocesses the image

- Trains a simple neural network

- Predicts handwritten digits

---

### **DATASET**

The dataset used is from the Kaggle Digit Recognizer competition, which is based on MNIST dataset. It consists of

- 60,000 training samples

- 10,000 testing samples

- 10 classes (digits 0-9)

The dataset is provided in a CSV format where:

- The first column represents the label (0-9)

- The remaining 784 columns represent the pixel intensity values (0-255), which are normalized to [0,1]

---

### **MODEL ARCHITECTURE**

The neural network consists of:

- Input Layer: 784 neurons (one for each pixel in the image; 28x28 = 784)

- Hidden Layer: 10 neurons with ReLU activation

- Output Layer: 10 neurons (one for each digit) with Softmax activation

**ACTIVATION FUNCTIONS USED**

- ReLU (Rectified Linear Unit): Used in the hidden layer to introduce non-linearity.

- Softmax: Used in the output layer to ensure the output is a probability distribution over the 10 digit classes (0-9).

---

### **THE MATHS BEHIND**

We start with 28 x 28 grayscale images, meaning each image has 784 pixels in it. Each pixel has an intensity value between 0 and 255 (where 0 = black and 255 = white).

**DATA REPRESENTATION**

Each image is flattened into a 784-element feature vector, forming a dataset matrix where:
- Each row represents an image

- Each column represents a pixel

This matrix is transposed, making each column an example rather than a row.

**NEURAL NETWORK COMPUTATION**

*Forward Propagation Steps*

1. Compute the weighted sum for the hidden layer: Z1 = W1 ⋅ X + b1

2. Apply the ReLU activation function to the hidden layer output: A1 = ReLU(Z1)

3. Compute the weighted sum for the output layer: Z2 = W2 ⋅ A1 + b2

4. Apply Softmax activation function to output layer: A2 = Softmax(Z2)

<br>*Backward Propagation Steps*

1. Compute gradients using cross-entropy loss.

2. Update weights and biases using gradient descent.

---

### **INSTALLATION AND DEPENDENCIES**

To run this project, you need:

- Python 3.x

- numpy

- pandas

- matplotlib

---

### **TRAINING THE MODEL**

The model is trained using gradient descent with a learning rate of 0.0001 for 1000 iterations. During training, accuracy updates are printed every 10 iterations.

---

### **RESULTS**

The model achieves an accuracy of approimately 85-90% on the validation set.
For better performance, consider:

- Adding more hidden layers

- Using CNNs

- Implementing an advanced otpimizer (Adam instead od basic gradient descent)

---

### **TROUBLESHOOTING**

1. **ValueError**: operands could not be broadcast together with shapes

- Ensure one-hot encoding is correctly implemented in the one_hot() function.

- Verify dimensions of X_train, Y_train, and A2 match properly.

<br>2. **Low accuracy (~10%)**

- Check if the dataset is properly normalized (0-1) before training.

- Verify weights and biases are correctly initialized.

- Try a higher learning rate (e.g., 0.1 instead of 0.0001).

---

### **REFERENCE**

- Samson Zhang's tutorial on building a neural network from scratch

---