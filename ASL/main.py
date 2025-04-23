import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the model
model = load_model("asl_model.h5")

# Define class names (same order as training generator's class_indices)
class_names = sorted(os.listdir('/home/pouya/workspace/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train'))

# Load and preprocess the image
img_path = "myownhand.jpg"
img = image.load_img(img_path, target_size=(64, 64))  # Use the same target size as during training
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]

print(f"Predicted ASL letter: {predicted_class}")



