import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(processed_data_dir):
    X = []
    y = []
    
    for speaker_folder in os.listdir(processed_data_dir):
        speaker_folder_path = os.path.join(processed_data_dir, speaker_folder)
        
        if os.path.isdir(speaker_folder_path):
            for mfcc_file in os.listdir(speaker_folder_path):
                if mfcc_file.endswith('.npy'):
                    mfcc_path = os.path.join(speaker_folder_path, mfcc_file)
                    mfcc = np.load(mfcc_path)
                    X.append(mfcc)
                    y.append(speaker_folder)  
    
    return np.array(X), np.array(y)

# Load the processed data
X, y = load_data(r"C:\Users\techg\OneDrive\Documents\Desktop\Voice Model\data\processed")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.save("data/train_X.npy", X_train)
np.save("data/train_y.npy", y_train)
np.save("data/test_X.npy", X_test)
np.save("data/test_y.npy", y_test)

print(f"Data split into training and testing sets. Training set size: {X_train.shape}, Testing set size: {X_test.shape}")
