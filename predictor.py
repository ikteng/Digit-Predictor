import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained model
model = load_model(r'Digit Recognizer\digit_recognition_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # check the background vs font
    avg_pixel_value = np.mean(grayscale_image)
    
    # Invert the colors if the image has a predominantly light background (dark font)
    if avg_pixel_value > 128:
        # Apply Otsu's thresholding to binarize the image
        _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Resize the binary image to 28x28 pixels (the same size as MNIST images)
        resized_image = cv2.resize(binary_image, (28, 28))
    else:
        # Resize the image to 28x28 pixels (the same size as MNIST images)
        resized_image = cv2.resize(grayscale_image, (28, 28))
    
    # Convert the image to a numpy array and normalize the pixel values
    image_array = resized_image / 255.0
    
    # Reshape the array to match the input shape of the model
    image_array = image_array.reshape(1, 28, 28)

    return image_array

# Function to predict the digit in the image
def predict_digit(image_path):
    # Preprocess the image
    image_array = preprocess_image(image_path)
    
    # Make predictions using the model
    predictions = model.predict(image_array)
    
    # Get the predicted digit (the index with the highest probability)
    predicted_digit = np.argmax(predictions)
    
    return predicted_digit

# Example usage:
image_path = r'Digit Recognizer\test\img_3.jpg'
predicted_digit = predict_digit(image_path)
print(f'The predicted digit in the image is: {predicted_digit}')
