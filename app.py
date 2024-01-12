from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model('best_model.h5')  # Load your trained model

    if 'file' not in request.files:
        return render_template('index.html', prediction="No file selected.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected.")

    img = image.load_img(file, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction, axis=1)

    # Map label to emotion (you need to define this mapping)
    emotion_mapping = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Neutral', 4: 'Surprise', 5: 'Fear', 6: 'Disgust'}
    predicted_emotion = emotion_mapping.get(predicted_label[0], 'Unknown')

    return render_template('index.html', prediction=f"Predicted Emotion: {predicted_emotion}")

if __name__ == '__main__':
    app.run(debug=True)
