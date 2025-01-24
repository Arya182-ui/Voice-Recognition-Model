# Speaker Recognition System

This project implements a Voice Recognition system that can identify speakers based on audio recordings. The system extracts Mel Frequency Cepstral Coefficients (MFCC) from raw audio, trains a Random Forest Classifier, and allows for real-time speaker identification.

## 🚀 Project Overview

The system consists of multiple components:

- **Data Preprocessing**: Extracts MFCC features from audio files using the librosa library.
- **Model Training**: Trains a Random Forest Classifier on the extracted features to recognize different speakers.
- **Prediction**: Allows for real-time prediction of speakers from newly recorded audio.
- **Data Collection**: Provides a simple interface to record audio and save it for training.

The entire project pipeline is designed for scalability and ease of deployment.

## ⚙️ Key Features

- **MFCC Extraction**: Uses librosa to preprocess raw audio files into MFCC features for better speaker recognition.
- **Random Forest Classifier**: Trains a model using scikit-learn to classify speakers based on their voice.
- **Real-time Recording & Prediction**: Enables recording new audio and predicting the speaker in real-time.
- **Modular Structure**: Easy-to-understand codebase, divided into logical scripts for data preparation, training, and prediction.

## 📂 Project Structure

The project is organized as follows:

voice-recognition/
├── data/                    # Raw and processed audio data
│   ├── raw/                 # Raw audio data
│   └── processed/           # Processed MFCC features
├── model/                   # Trained models
│   └── speaker_model.pkl    # Random forest model for speaker recognition
├── src/                     # Source code for the project
│   ├── data_preprocessing.py  # Script to preprocess raw audio files
│   ├── model_training.py      # Script to train the Random Forest Classifier
│   ├── predict_speaker.py     # Script to predict speakers from new audio
│   ├── prepare_data.py        # Script to prepare data for training
│   ├── record.py              # Script to record new audio
│   └── utils.py               # Utility functions
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation


## 🛠️ Installation

1. Clone the repository:

```bash
   git clone https://github.com/your-username/voice-recognition.git
   
   cd voice-recognition

   ```

``` bash
   pip install -r requirements.txt
```

## Run the Scripts 

```bash
   src/data_preprocessing.py

   src/model_training.py

   src/predict_speaker.py
```

## 📜 Data Collection
You can record your own audio files to add to the dataset for training. Use the src/record.py script to record a series of audio files for a new speaker. 
(For More knowldge create the record file by yourself otherwise if there is trouble then Meassege me )
save the audio recordings in the data/raw/{speaker_name} directory.

## 🔄 Training the Model
Once you have collected enough audio data, run the src/prepare_data.py to process the data and split it into training and testing sets.

```bash
python src/prepare_data.py
```
Then, use src/model_training.py to train the Random Forest Classifier on the processed features.

```bash
python src/model_training.py
```
The trained model will be saved as model/speaker_model.pkl.

## 🎤 Real-Time Prediction
To predict the speaker from a real-time recording, use the src/predict_speaker.py script. This script will:

* Record audio using your microphone
* Extract MFCC features from the recording
* Predict the speaker using the pre-trained model

```bash
python src/predict_speaker.py
```

## 🔧 Requirements
Python 3.7+
librosa for audio processing
scikit-learn for machine learning
numpy for numerical operations
sounddevice for real-time audio recording
joblib for saving the trained model


```bash
pip install -r requirements.txt
```

## 📝 Contributing
Feel free to fork this repository and submit pull requests. I welcome improvements, suggestions, and contributions from the community.

## 📄 License
This project is open-source and available under the MIT License.
