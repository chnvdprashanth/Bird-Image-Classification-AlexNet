# Bird Image Classifier

## Table of Contents
- [Screenshots](#screenshots)
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Future Scope](#future-scope)
- [Contributing](#contributing)
- [License](#license)

## Screenshots
Here are some screenshots of the Streamlit app in action:

### Home Page
![Home Page](https://drive.usercontent.google.com/download?id=1R74RIeLu0UauDE9pxn2UluRBWDsmgOJU&authuser=0)

### Upload Image
![Upload Image](https://drive.usercontent.google.com/download?id=1xFM93_i8gr3oXrl9Fx3FhlM5xxza64bO&authuser=0)

### Prediction Result
![Prediction Result](https://drive.usercontent.google.com/download?id=1qwuWDseVABPbPWbSvqOP1NMThB4AW1AL&authuser=0)

## Introduction
The Bird Image Classifier is a machine learning project inspired by the AlexNet architecture. It is designed to classify bird species from images. The model is trained on a dataset of 20 bird species obtained from Kaggle and features a user interface built with Streamlit.

## Features
- Classification of 20 bird species
- User interface using Streamlit
- High accuracy on test data
- Handles images of size 224x224 pixels

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/bird-image-classifier.git
    ```
2. Navigate to the project directory:
    ```bash
    cd bird-image-classifier
    ```
3. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To use the Bird Image Classifier, follow these steps:

1. Ensure you have the model file (`bird_image_classification.h5`) in the project directory.
2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3. Open your browser and go to `http://localhost:8501`.
4. Upload an image of a bird (224x224 pixels) and get the predicted bird species.

## Model Training
The model is trained using TensorFlow and Keras. Key details about the training process:

- **Architecture:** Inspired by AlexNet
- **Dataset:** 20 Bird Species from Kaggle
- **Libraries:** TensorFlow, Keras, Pandas, NumPy, Streamlit, Matplotlib
- **Train Accuracy:** ~87%
- **Validation Accuracy:** ~89%

### Data Preparation
The dataset is divided into training, validation, and test sets. Images are loaded and preprocessed to ensure they are of size 224x224 pixels.

### Model Architecture
The model architecture includes several convolutional layers followed by max-pooling and dropout layers to prevent overfitting. The final layers are fully connected with a softmax activation function to classify the 20 bird species.

### Training Process
The model is trained using the Adam optimizer and categorical cross-entropy loss function. Early stopping is used to prevent overfitting. The training process involves data augmentation to improve the model's generalization ability.

### Visualization
The notebook includes visualizations of random images from the training dataset along with their corresponding labels to provide insights into the data.

## Future Scope
This project currently classifies 20 bird species, but it can be extended to classify many more species. The dataset used is a subset of a larger dataset available on Kaggle, which contains around 1300 bird species. Future work could involve:

- Expanding the model to classify all 1300 bird species by retraining it on the larger dataset.
- Improving the model accuracy by experimenting with different architectures and hyperparameters.
- Incorporating transfer learning with pre-trained models like ResNet to improve performance.
- Enhancing the user interface for better user experience and additional features such as bird species information.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
