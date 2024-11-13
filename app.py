import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np

# Define constants
DATASET_PATH = "dataset"  # Directory containing A.wav to Z.wav
SAMPLE_RATE = 16000
DURATION = 1.0
NUM_CLASSES = 26  # A to Z
N_MFCC = 40  # Increased number of MFCC features for better representation

# Mapping each letter to an integer label
letter_to_label = {chr(i + 65): i for i in range(26)}  # A -> 0, B -> 1, ..., Z -> 25
label_to_letter = {i: chr(i + 65) for i in range(26)}

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the LSTM model structure with dropout
class SpellingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpellingRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Bidirectional doubles hidden size

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), hidden_size).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(2, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the output from the last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out  # Shape: (batch_size, output_size)

# Load and preprocess the audio data with MFCCs
def load_data_with_mfcc(dataset_path, sample_rate, duration, n_mfcc=N_MFCC):
    data = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".wav"):
            letter = filename.split(".")[0].upper()
            if letter in letter_to_label:
                file_path = os.path.join(dataset_path, filename)
                waveform, _ = librosa.load(file_path, sr=sample_rate, duration=duration)
                # Ensure waveform length is consistent
                if len(waveform) < int(sample_rate * duration):
                    pad_length = int(sample_rate * duration) - len(waveform)
                    waveform = np.pad(waveform, (0, pad_length), mode='constant')
                elif len(waveform) > int(sample_rate * duration):
                    waveform = waveform[:int(sample_rate * duration)]
                mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
                # Normalize MFCCs
                mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
                mfcc = torch.tensor(mfcc, dtype=torch.float32).transpose(0, 1)  # Shape: (T, n_mfcc)
                label = letter_to_label[letter]
                data.append(mfcc)
                labels.append(label)
    return data, labels

# Prepare dataset using MFCC
data, labels = load_data_with_mfcc(DATASET_PATH, SAMPLE_RATE, DURATION)
if not data:
    print("No audio files found in the dataset path.")
    exit()
data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True).to(device)  # Pad sequences
labels = torch.tensor(labels).to(device)

# Model parameters
input_size = N_MFCC
hidden_size = 128
output_size = NUM_CLASSES

# Instantiate model, loss function, and optimizer
model = SpellingRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(data)  # Shape: (batch_size, output_size)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0) * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "spelling_rnn_model.pth")
print("Model saved as spelling_rnn_model.pth")
