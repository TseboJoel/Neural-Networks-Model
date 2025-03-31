# Traffic Sign Recognition Using Deep Learning

## Overview

This project trains a deep learning model to recognize traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. It uses TensorFlow/Keras for model training and Tkinter for building a simple GUI to test the trained model with uploaded images.

## Dataset
The dataset used for training is GTSRB (German Traffic Sign Recognition Benchmark), which contains 43 categories of traffic signs. Each category represents a different type of sign, such as speed limits, warnings, and prohibitions.

## Installation
Before running the project, ensure you have Python installed. Then install the required dependencies:

## Getting Started with the Training / using the model
To train the model, run the following command:

### Prerequisites
- Python
- TensorFlow
- Pillow
- Numpy
- sklearn
- Anaconda

1. Clone the repository in your anaconda prompt
```bash
git clone https://github.com/TseboJoel/Neural-Networks-Model.git
cd Neural-Networks-Model
```

2. activate tensorflow environment
```bash
conda activate tf
```

4. install the dependencies and packages
```bash
pip install pillow numpy sklearn
```

5. Train the model
```bash
cd TSEBO
python traffic.py path/to/data_directory path/to/model.h5
```

6. Use the model
Still within TSEBO directory
```bash
python predict_sign.py path/to/model
```

## Training Configuration
- Number of Epochs: You can adjust the number of epochs in train.py to optimize performance.

- Test Size: The dataset is split using train_test_split(test_size=0.2), meaning 80% of the data is used for training and 20% for testing.

- Image Normalization: Images are resized and normalized (image / 255.0) for better training performance.


## Understanding traffic.py
- load_data(data_dir)
Loads and preprocesses images from data_dir.

Resizes images to (IMG_WIDTH, IMG_HEIGHT).

Normalizes pixel values to [0, 1].

- get_model()
Builds and returns a CNN model using TensorFlow/Keras.

Contains:

Convolutional layers (Conv2D)

Pooling layers (MaxPooling2D)

Fully connected (Dense) layers

Dropout for regularization

## Improving the Model
To enhance model performance, experiment with:

- Number of Filters: More filters extract better features.

- Kernel Size: Larger kernels capture more spatial relationships.

- Dropout Rate: Prevents overfitting.

- Number of Epochs: Adjust based on dataset size.

## Evaluation Metrics
Training and validation accuracy are tracked during training.

Final accuracy is measured using model.evaluate().

Confusion matrices and precision-recall scores can be used for deeper analysis.