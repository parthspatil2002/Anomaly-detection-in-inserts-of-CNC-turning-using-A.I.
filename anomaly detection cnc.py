# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

# Define paths for training and validation data
train_data_dir = 'path/to/your/training/images'
validation_data_dir = 'path/to/your/validation/images'

# Create data generators for image augmentation (optional)
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Define image dimensions and batch size
img_width, img_height = 150, 150  # Adjust as needed based on your image size
batch_size = 32

# Use data generators to load images and labels
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'  # One-hot encode labels
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))  # Adjust the number of hidden units as needed
model.add(Dense(5, activation='softmax'))  # 5 for 5 defect classes (modify based on your classes)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs for better training
    validation_data=validation_generator
)

# Save the model (optional)
model.save('insert_defect_classifier.h5')

# Function to predict defect on a new image (to be implemented after training)
def predict_defect(image_path):
  # Load the image and preprocess it
  img = ...  # Load and preprocess the image based on your data format
  img = np.expand_dims(img, axis=0)  # Add a batch dimension

  # Make prediction
  prediction = model.predict(img)
  defect_class = np.argmax(prediction[0])  # Get the index of the most likely class

  # Map the index to the actual defect name based on your class labels
  defect_name = {0: 'Flank Wear', 1: 'Crater Wear', 2: 'Nose Wear', 3: 'Chipping', 4: 'Built-Up Edge'}[defect_class]  # Modify class names based on yours

  return defect_name

# Example usage (after training)
defect_name = predict_defect('path/to/your/new/image.png')
print(f'Predicted Defect: {defect_name}')
