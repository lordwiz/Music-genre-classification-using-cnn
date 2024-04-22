#Music Genre Classification using CNN
This project aims to classify music genres using spectrogram images generated from audio files. It utilizes Convolutional Neural Networks (CNNs) to learn features from the spectrogram images and predict the genre of the music.

#Features
 -Converts audio files to spectrogram images
 -Utilizes MinMaxScaler for feature scaling
 -Resizes spectrogram images to match the input size of the CNN model
 -Predicts music genre using a trained CNN model
 -Supports various music genres including Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, and Rock
#Requirements
 -Python 3.x
 -numpy
 -librosa
 -matplotlib
 -cv2
 -tensorflow
 -scikit-learn
#Installation
**Clone the repository:**
'''
git clone https://github.com/username/music-genre-classification.git
'''
#Install the required dependencies:
'''
pip install -r requirements.txt
'''
#Usage
Place your audio files in the specified directory.
Run the script classify_music.py to generate spectrogram images and predict the music genre:
'''
python classify_music.py
'''
The predicted genre for each audio file will be displayed in the console.
#Model
The classification model used in this project is a pre-trained CNN model trained on a large dataset of spectrogram images.

#Dataset
The dataset used for training and testing is "GTZAN Dataset - Music Genre Classification", the model consists of audio files from various music genres. Each audio file is manually labeled with its corresponding genre.
