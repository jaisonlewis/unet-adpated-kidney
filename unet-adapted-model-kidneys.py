import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

BACKBONE = 'seresnext50' #change this to change the backbone
preprocess_input = sm.get_preprocessing(BACKBONE)

# Resizing images is optional (improves speed while testing backbones)
SIZE_X = 256  # Resize images (height = X, width = Y)
SIZE_Y = 256  # Resnet etc take 224, inception works best with 299, image size is 512

# Function to read and preprocess images and masks
def read_and_preprocess_images_masks(directory, size_x, size_y):
    images = []
    image_names = []
    for img_path in glob.glob(os.path.join(directory, "*.png")):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size_y, size_x))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = img / 255.0 # Normalize pixel values between 0 and 1
        images.append(img)
        image_names.append(img_name)
    images = np.array(images)
    return images, image_names

def read_and_preprocess_masks(directory, size_x, size_y, image_names):
    masks = []
    for img_name in image_names:
        mask_path = os.path.join(directory, img_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (size_y, size_x))
        mask = np.expand_dims(mask, axis=2)  # Add channel dimension
        masks.append(mask)
    masks = np.array(masks)
    return masks

# Capture training image info as a list
train_images, train_image_names = read_and_preprocess_images_masks("kidney_train", SIZE_X, SIZE_Y)

# Capture mask/label info as a list
train_masks = read_and_preprocess_masks("kidney_mask", SIZE_X, SIZE_Y, train_image_names)

# Check if the number of images and masks match
if len(train_images) != len(train_masks):
    raise ValueError("Number of images and masks do not match!")

# Use customary x_train and y_train variables
X = train_images
Y = train_masks


# Check if Y (train_masks) is not empty
if len(Y) == 0:
    raise ValueError("No grayscale masks found in the 'kidney_mask' directory!")

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

print(model.summary())

# Add Early Stopping
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Add Model Checkpoint
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_kidney_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(x_train, y_train, batch_size=8, epochs=100, verbose=1, validation_data=(x_val, y_val),
                    callbacks=[model_checkpoint]) #add earlystopping here (early_stopping,)

# Plot the training and validation loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load the best model
best_model = tf.keras.models.load_model('best_kidney_model.h5')

# Visualization: Show sample input images, ground truth masks, and model predictions
n_samples = 5
sample_indices = np.random.randint(0, len(x_val), n_samples)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(sample_indices):
    plt.subplot(3, n_samples, i + 1)
    plt.imshow(x_val[idx])
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(3, n_samples, n_samples + i + 1)
    plt.imshow(np.squeeze(y_val[idx]), cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(3, n_samples, 2 * n_samples + i + 1)
    predicted_mask = best_model.predict(np.expand_dims(x_val[idx], axis=0))
    plt.imshow(np.squeeze(predicted_mask), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

plt.tight_layout()
plt.show()
