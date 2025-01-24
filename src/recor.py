import sounddevice as sd
import numpy as np
import os
import wave
import time

# Define parameters
SAMPLE_RATE = 16000  
CHANNELS = 1 
DURATION = 5  
NUM_RECORDINGS = 30 

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=CHANNELS, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return audio.flatten()


def save_audio(filename, audio_data, sample_rate=SAMPLE_RATE):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2) 
        wf.setframerate(sample_rate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":

    speaker_name = input("Enter the name of the speaker (e.g., speaker_1): ")
    

    output_dir = f"data/raw/{speaker_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Recording {NUM_RECORDINGS} audio files for {speaker_name}...")


    for i in range(1, NUM_RECORDINGS + 1):
        filename = os.path.join(output_dir, f"{speaker_name}_audio_{i}.wav")
        print(f"Recording file {i}/{NUM_RECORDINGS}...")
        

        audio = record_audio(duration=DURATION)
        

        save_audio(filename, audio)

        if i < NUM_RECORDINGS:  # Don't wait after the last recording
            print(f"Recording {i} saved as {filename}. Preparing for the next recording...")
            time.sleep(1) 
    
    print(f"Recording complete. {NUM_RECORDINGS} files saved in {output_dir}.")
