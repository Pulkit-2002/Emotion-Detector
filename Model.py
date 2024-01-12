import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os

# Model Selection

# a. Choose Architecture
def create_cnn_model(input_shape=(48, 48, 3), num_classes=7):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# b. Pre-trained Models
def fine_tune_pretrained_model(base_model, num_classes=7):
    # Freeze the layers of the pre-trained model
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Example usage
input_shape = (48, 48, 3)
num_classes = 7

# Create a CNN model
cnn_model = create_cnn_model(input_shape, num_classes)

# Download a pre-trained model (VGG16 in this example)
pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Fine-tune the pre-trained model
fine_tuned_model = fine_tune_pretrained_model(pretrained_model, num_classes)

# Display model summaries
print("CNN Model Summary:")
cnn_model.summary()

print("\nFine-tuned Pre-trained Model Summary:")
fine_tuned_model.summary()

# Model Training

# a. Data Splitting
def split_data(dataset_path, validation_split=0.2, test_split=0.1):
    # Get the list of all images in the dataset
    all_images = [os.path.join(dataset_path, emotion, image) for emotion in os.listdir(dataset_path)
                  for image in os.listdir(os.path.join(dataset_path, emotion))]

    # Split the data into training, validation, and test sets
    train_images, test_images = train_test_split(all_images, test_size=test_split, random_state=42)
    train_images, val_images = train_test_split(train_images, test_size=validation_split, random_state=42)

    return train_images, val_images, test_images

# Example usage
dataset_path = 'path_to_dataset'
train_images, val_images, test_images = split_data(dataset_path)

# b. Training Process
def train_model(model, train_generator, val_generator, epochs=10, batch_size=32):
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        batch_size=batch_size
    )

    return model, history

# Example usage
# Assuming you have a data generator for training and validation images (train_generator, val_generator)

trained_model, training_history = train_model(cnn_model, train_generator, val_generator)

# c. Transfer Learning
def fine_tune_model(model, train_generator, val_generator, base_model, fine_tune_epochs=5, epochs=10, batch_size=32):
    # Freeze early layers of the base model
    base_model.trainable = False

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train only the later layers
    history_fine_tune = model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        validation_data=val_generator,
        batch_size=batch_size
    )

    # Unfreeze all layers for further training
    base_model.trainable = True

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the entire model
    history_full_train = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        batch_size=batch_size
    )

    return model, history_fine_tune, history_full_train

# Example usage
# Assuming you have a data generator for training and validation images (train_generator, val_generator)
# Also, assuming you have a fine-tuned model (fine_tuned_model) from the previous step

fully_trained_model, fine_tune_history, full_train_history = fine_tune_model(
    fine_tuned_model,
    train_generator,
    val_generator,
    pretrained_model
)
