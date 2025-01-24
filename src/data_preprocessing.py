import librosa
import numpy as np
import os

# Define constants
SAMPLE_RATE = 16000  # 16 kHz
N_MFCC = 13         
DURATION = 5         
N_FFT = 2048         # Number of FFT components
HOP_LENGTH = 512  
OUTPUT_DIR = r"C:\Users\techg\OneDrive\Documents\Desktop\Voice Model\data\processed"  

# Function to extract MFCC features from an audio file
def extract_mfcc(audio_path):
    """
    Extract MFCC features from a .wav file.

    Args:
        audio_path (str): Path to the .wav audio file.

    Returns:
        np.ndarray: Mean of the MFCC coefficients across time frames.
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.mean(mfcc, axis=1)

# Function to process all audio files in a directory and save their MFCC features
def process_data(input_dir, output_dir):
    """
    Process all .wav files in the input directory to extract MFCC features and save them.

    Args:
        input_dir (str): Directory containing raw audio files (e.g., data/raw).
        output_dir (str): Directory to save processed MFCC features (e.g., data/processed).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for speaker_folder in os.listdir(input_dir):
        speaker_folder_path = os.path.join(input_dir, speaker_folder)
        
        if os.path.isdir(speaker_folder_path):
        
            speaker_output_dir = os.path.join(output_dir, speaker_folder)
            os.makedirs(speaker_output_dir, exist_ok=True)
            
            for audio_file in os.listdir(speaker_folder_path):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(speaker_folder_path, audio_file)
                    mfcc = extract_mfcc(audio_path)
                    np.save(os.path.join(speaker_output_dir, f"{audio_file.split('.')[0]}.npy"), mfcc)
                    print(f"Processed and saved MFCC for: {audio_file}")
    print("Data processing complete.")

# Run the preprocessing
if __name__ == "__main__":
    process_data(r"C:\Users\techg\OneDrive\Documents\Desktop\Voice Model\data\raw", OUTPUT_DIR)
