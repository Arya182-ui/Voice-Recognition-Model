import numpy as np
import librosa
import joblib
import sounddevice as sd
import os
import wave

# Load the pre-trained model
model = joblib.load(r"C:\Users\techg\OneDrive\Documents\Desktop\Voice Model\model\speaker_model.pkl")

# Define parameters for MFCC extraction
SAMPLE_RATE = 16000  # 16 kHz
N_MFCC = 13          
N_FFT = 2048       
HOP_LENGTH = 512     

# Function to extract MFCC from an audio file
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.mean(mfcc, axis=1)

# Function to record audio
def record_audio(duration=5, sample_rate=SAMPLE_RATE, filename="recorded_audio.wav"):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")

    # Save the audio data as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1) 
        wf.setsampwidth(2)  
        wf.setframerate(sample_rate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes()) 

    return filename 

# Function to predict the speaker
def predict_speaker(audio_path):
    mfcc = extract_mfcc(audio_path)
    
    mfcc = mfcc.reshape(1, -1)
    
    prediction = model.predict(mfcc)
    
    return prediction[0]

# Main logic
if __name__ == "__main__":

    audio_path = record_audio()
    
    speaker = predict_speaker(audio_path)
    
    print(f"Predicted speaker: {speaker}")
    
    # Remove Audio Fille After Predict Speaker
    os.remove(audio_path)
    print(f"Deleted the recorded audio file: {audio_path}")
