import cv2
from flask import Flask, render_template, request, flash
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)
app.secret_key = "11"

model = load_model('imageclass.h5')  # Replace with the actual path to your model
class_labels = ['Blue', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    return render_template('login.html')

@app.route('/check_login', methods=['POST'])
def check_login():
    database = {'admin': 'admin'}
    uname = request.form['uname']
    passw = request.form['pass']

    if uname not in database or database[uname] != passw :
        return render_template('login.html', info='WRONG USERNAME OR PASSWORD')
    
    return render_template('prediction.html', info='WELCOME TO MUSIC CLASSIFIER')

@app.route('/prediction', methods=['POST', 'GET'])
def phome():
    return render_template('prediction.html', info='WELCOME TO MUSIC CLASSIFIER')

@app.route('/predict_music', methods=['POST'])
def predict_music():
    aud = request.files['aud']
    aud_path = './audios/' + 'uploaded.wav'
    aud.save(aud_path)


# Load the external audio file
    y, sr = librosa.load(aud_path)

    spectrogram = np.abs(librosa.stft(y))
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)  # Convert to dB scale
    spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add a channel dimension
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add a batch dimension


    scaler = MinMaxScaler(feature_range=(0, 1))
    spectrogram_scaled = scaler.fit_transform(spectrogram[0, :, :, 0])

# Reshape to match the target shape
    spectrogram_scaled = spectrogram_scaled.reshape(1, spectrogram_scaled.shape[0], spectrogram_scaled.shape[1], 1)

# Resize the entire spectrogram using cv2.resize
    target_shape = model.input_shape[1:-1]
    dsize = (target_shape[1], target_shape[0])  # Ensure non-zero values and swap height and width
    spectrogram_resized = cv2.resize(spectrogram_scaled[0, :, :, 0], dsize, interpolation=cv2.INTER_LINEAR)
    spectrogram_resized = np.expand_dims(spectrogram_resized, axis=-1)  # Add the channel dimension back
    spectrogram_resized = np.repeat(spectrogram_resized, 3, axis=-1)

# ... (rest of your model prediction code)

# Make predictions using the loaded model
    predictions = model.predict(np.expand_dims(spectrogram_resized, axis=0))

# Get the predicted label
    predicted_label = class_labels[np.argmax(predictions)]

    return render_template('prediction.html', Predicted=f'Predicted Label: {predicted_label}')

if __name__ == '__main__':
    app.run(debug=True)
