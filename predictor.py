import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained model
model = load_model(r"Digit Predictor\digit_recognition_model.h5")

# Function to preprocess the image and extract bounding boxes
def preprocess_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check the background vs font
    avg_pixel_value = np.mean(grayscale_image)
    
    # Invert the colors if the image has a predominantly light background (dark font)
    if avg_pixel_value > 128:
        # Apply Otsu's thresholding to binarize the image
        _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary_image = grayscale_image
    
    # Find connected components and retrieve bounding boxes
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)
    
    digit_images = []
    bounding_boxes = []
    
    # Loop over the connected components
    for stat in stats[1:]:  # Exclude background component
        # Extract bounding box coordinates
        x, y, w, h = stat[0], stat[1], stat[2], stat[3]
        
        # Extract the digit region from the original image
        digit_region = binary_image[y:y+h, x:x+w]
        
        # Resize the digit region to 28x28 pixels
        digit_region_resized = cv2.resize(digit_region, (28, 28))
        
        # Normalize the pixel values
        digit_region_normalized = digit_region_resized / 255.0
        
        # Reshape the array to match the input shape of the model
        digit_image = digit_region_normalized.reshape(1, 28, 28)
        
        digit_images.append(digit_image)
        bounding_boxes.append(((x, y, w, h), digit_image))  # Store bounding box coordinates and digit image
    
    return digit_images, bounding_boxes, image, binary_image  # Return digit images, bounding boxes, original image, and binary image

# Function to predict the digits in the image
def predict_digits(image_path):
    # Preprocess the image to extract individual digits and bounding boxes
    digit_images, bounding_boxes, original_image, binary_image = preprocess_image(image_path)

    # Sort bounding boxes by y-coordinate
    bounding_boxes.sort(key=lambda box: box[0][1])

    # Split bounding boxes into two rows based on y-coordinate
    rows = [[], []]
    for box in bounding_boxes:
        if box[0][1] < binary_image.shape[0] / 2:  # Assuming the two rows are separated approximately at the middle
            rows[0].append(box)
        else:
            rows[1].append(box)

    # Sort digits within each row by x-coordinate
    for row in rows:
        row.sort(key=lambda box: box[0][0])

    # Concatenate predicted digits from both rows
    predicted_digits = []
    for row in rows:
        for _, digit_image in row:
            # Make predictions using the model
            predictions = model.predict(digit_image)
            # Get the predicted digit (the index with the highest probability)
            predicted_digit = np.argmax(predictions)
            predicted_digits.append(predicted_digit)

    return predicted_digits, original_image

# Example usage:
image_path = r'Digit Predictor\test\img_6.jpg'
predicted_digits, image_with_boxes = predict_digits(image_path)
print(f'The predicted digits in the image are: {predicted_digits}')

# # Display the image with bounding boxes
# cv2.imshow('Image with Bounding Boxes', image_with_boxes)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

