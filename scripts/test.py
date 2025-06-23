import torch
import numpy as np
import librosa
import os 

# Emotion mapping (same as training)
EMOTION_DICT = {
    'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
    'angry': 4, 'fear': 5, 'disgust': 6, 'surprise': 7
}
REVERSE_EMOTION = {v: k for k, v in EMOTION_DICT.items()}

# Define the CNNLSTM model class
class CNNLSTM(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Dropout(0.25),
            
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Dropout(0.25),
            
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Dropout(0.3),
            
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 1)),
            torch.nn.Dropout(0.3)
        )
        self.lstm_input_size = 256 * 9
        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return torch.nn.functional.log_softmax(out, dim=1)

def extract_features(file_path):
    """Extract audio features same as training pipeline"""
    data, sample_rate = librosa.load(file_path, duration=2.5, offset=0.6)
    
    # Feature extraction
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=data))
    
    return np.hstack([chroma, mfcc, zcr, mel, rms])

def predict_emotion(audio_path, model_path='models/best_model.pth'):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTM(num_classes=8).to(device)
    
    # Load state dictionary directly
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Extract and format features
    features = extract_features(audio_path)
    features_tensor = torch.tensor(features.reshape(1, 1, 9, 18), 
                                  dtype=torch.float32).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return REVERSE_EMOTION[predicted.item()]

if __name__ == "__main__":
    audio_path = input("Enter the path to the audio file: ")
    model_path =  "models/best_model.pth"

    try:
        emotion = predict_emotion(audio_path, model_path)
        print(f"\nPredicted emotion: {emotion}")
    except FileNotFoundError:
        print("Error: File not found. Please check the path and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")
