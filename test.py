import os
import torch
import torch.nn as nn
import librosa
import numpy as np

# Define constants
TEST_DATASET_PATH = "test_dataset"  # Directory containing test audio files
SAMPLE_RATE = 16000
DURATION = 1.0
NUM_CLASSES = 26  # A to Z
N_MFCC = 40  # Should match the training configuration

# Mapping each integer label to a letter
label_to_letter = {i: chr(i + 65) for i in range(26)}  # 0 -> 'A', 1 -> 'B', ..., 25 -> 'Z'

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the LSTM model structure with dropout (must match training)
class SpellingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpellingRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Bidirectional doubles hidden size

    def forward(self, x):
        h0 = torch.zeros(
            2, x.size(0), hidden_size
        ).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(2, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[
            :, -1, :
        ]  # Take the output from the last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out  # Shape: (batch_size, output_size)

# Load and preprocess the audio data with MFCCs
def load_data_with_mfcc(dataset_path, sample_rate, duration, n_mfcc=N_MFCC):
    data = []
    file_names = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(dataset_path, filename)
            waveform, _ = librosa.load(file_path, sr=sample_rate, duration=duration)
            # Ensure waveform length is consistent
            expected_length = int(sample_rate * duration)
            if len(waveform) < expected_length:
                pad_length = expected_length - len(waveform)
                waveform = np.pad(waveform, (0, pad_length), mode="constant")
            elif len(waveform) > expected_length:
                waveform = waveform[:expected_length]
            mfcc = librosa.feature.mfcc(
                y=waveform, sr=sample_rate, n_mfcc=n_mfcc
            )
            # Normalize MFCCs
            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
            mfcc = torch.tensor(mfcc, dtype=torch.float32).transpose(
                0, 1
            )  # Shape: (T, n_mfcc)
            data.append(mfcc)
            file_names.append(filename)
    return data, file_names

# Prepare test dataset
data, file_names = load_data_with_mfcc(
    TEST_DATASET_PATH, SAMPLE_RATE, DURATION
)
if not data:
    print("No audio files found in the test dataset path.")
    exit()
data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True).to(device)

# Model parameters (must match training)
input_size = N_MFCC
hidden_size = 128
output_size = NUM_CLASSES

# Instantiate and load the model
model = SpellingRNN(input_size, hidden_size, output_size).to(device)
model.load_state_dict(
    torch.load("spelling_rnn_model.pth", map_location=device)
)
model.eval()
print("Model loaded.")

# Perform inference
with torch.no_grad():
    outputs = model(data)
    _, predicted = torch.max(outputs.data, 1)

# Output predictions
print("Predictions:")
for i in range(len(file_names)):
    predicted_letter = label_to_letter[predicted[i].item()]
    print(f"File: {file_names[i]}, Predicted: {predicted_letter}")
