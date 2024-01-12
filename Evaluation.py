import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Evaluation

# a. Classification Metrics
def evaluate_model(model, test_generator):
    # Predict on the test set
    predictions = model.predict(test_generator)

    # Convert predictions to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Get true labels
    true_labels = test_generator.classes

    # Classification Report
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plot_confusion_matrix(cm, test_generator.class_indices.keys())

# Helper function to plot the confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Example usage
# Assuming you have a data generator for testing images (test_generator)

evaluate_model(trained_model, test_generator)

# b. Real-world Testing
def real_world_testing(model, image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(48, 48))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values to be between 0 and 1

    # Make predictions
    predictions = model.predict(img_array)

    # Convert predictions to class labels
    predicted_label = np.argmax(predictions, axis=1)

    # Get the class name based on the predicted label
    class_names = {value: key for key, value in train_generator.class_indices.items()}
    predicted_class_name = class_names[predicted_label[0]]

    return predicted_class_name

# Example usage
# Assuming you have a real-world image file (e.g., 'path_to_real_world_image.jpg')
real_world_image_path = 'path_to_real_world_image.jpg'
predicted_class = real_world_testing(trained_model, real_world_image_path)
print(f"Predicted Emotion for Real-world Image: {predicted_class}")
