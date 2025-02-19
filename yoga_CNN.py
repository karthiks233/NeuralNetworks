from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Function to check for corrupted images
def check_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        img.close()
        return True
    except Exception as e:
        print(f"Corrupted image: {file_path} - {e}")
        return False

# Directory containing images
image_dir = "DATASET/TRAIN/plank/"

# Check and remove corrupted images
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        file_path = os.path.join(image_dir, filename)
        if not check_image(file_path):
            os.remove(file_path)  # Remove corrupted images


# Directory containing images
image_di = "DATASET/TEST/plank/"

# Check and remove corrupted images
for filename in os.listdir(image_di):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        file_pat = os.path.join(image_di, filename)
        if not check_image(file_pat):
            os.remove(file_pat)  # Remove corrupted images


# Define paths to the dataset
train_dir = "DATASET/TRAIN/"
validation_dir = "DATASET/TEST/"

# Image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
)

# Debug: Print the number of images found
print(f"Training images found: {train_generator.samples}")
print(f"Validation images found: {validation_generator.samples}")

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Print model summary
model.summary()

# # Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
#     epochs=20,
#     validation_data=validation_generator,
#     validation_steps=max(1, validation_generator.samples // validation_generator.batch_size),
# )

# # Plot training and validation accuracy
# plt.plot(history.history["accuracy"], label="Training Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# # Plot training and validation loss
# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# # Save the model
# model.save("plank_pose_model.h5")

# # Test the model on a new image
# img_path = "DATASET/TEST/plank/00000000.jpg"
# if os.path.exists(img_path):
#     img = Image.open(img_path)
#     img = img.resize((150, 150))  # Resize the image
#     img_array = np.array(img)  # Convert to numpy array
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array /= 255.0  # Normalize pixel values

#     # Make a prediction
#     prediction = model.predict(img_array)
#     if prediction[0] > 0.5:
#         print("Plank Pose Detected!")
#     else:
#         print("Not a Plank Pose.")
# else:
#     print(f"Error: Test image not found at {img_path}")