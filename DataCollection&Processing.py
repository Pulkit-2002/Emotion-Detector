import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_file
import cv2
import dlib
import os

# Data Collection

# a. Image Datasets
def download_ckplus_dataset():
    base_url = 'http://www.pitt.edu/~emotion/ck-spread/'
    archive_name = 'ckplus.zip'
    data_dir = get_file(archive_name, base_url + archive_name, extract=True)
    return os.path.join(os.path.dirname(data_dir), 'Emotion', 'sub')

# Example usage for CK+ dataset
ckplus_dataset_path = download_ckplus_dataset()

# b. Data Augmentation
def augment_data(dataset_path, save_path, batch_size=32, target_size=(48, 48)):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    dataset_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        save_to_dir=save_path,
        save_prefix='aug_',
        save_format='png'
    )

    return dataset_generator

# Example usage for data augmentation
augmented_data_path = 'path_to_save_augmented_data'
augmentation_generator = augment_data(ckplus_dataset_path, augmented_data_path)

# Display augmented images (for illustration purposes)
import matplotlib.pyplot as plt

# Only display the first 5 augmented images
for i in range(5):
    augmented_image, _ = augmentation_generator.next()
    plt.imshow(augmented_image[0].astype('uint8'))
    plt.show()

# Data Preprocessing

# a. Face Detection
def detect_faces(image_path, face_cascade_path='haarcascade_frontalface_default.xml'):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Extract faces from the image
    face_images = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_images.append(face)

    return face_images

# Example usage for face detection
image_path = 'path_to_image.jpg'
detected_faces = detect_faces(image_path)

# b. Facial Landmarks
def detect_facial_landmarks(image_path, shape_predictor_path='shape_predictor_68_face_landmarks.dat'):
    # Load the pre-trained facial landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Extract facial landmarks for each detected face
    facial_landmarks = []
    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]  # Assumes 68 landmarks
        facial_landmarks.append(landmarks)

    return facial_landmarks

# Example usage for facial landmarks detection
landmarks = detect_facial_landmarks(image_path)

# c. Image Resizing
def resize_images(images, target_size=(48, 48)):
    resized_images = [cv2.resize(image, target_size) for image in images]
    return resized_images

# Example usage for image resizing
resized_faces = resize_images(detected_faces)

# Display the results (for illustration purposes)
import matplotlib.pyplot as plt

# Original image
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Detected faces
for i, face in enumerate(detected_faces):
    plt.subplot(1, len(detected_faces), i+1)
    plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Face {i+1}')
plt.show()

# Resized faces
for i, face in enumerate(resized_faces):
    plt.subplot(1, len(resized_faces), i+1)
    plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    plt.title(f'Resized Face {i+1}')
plt.show()
