import streamlit as st
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import os 
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    .title {
        color: #1e3a8a;
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* Upload box styling */
    .upload-box {
        border: 2px dashed #4f46e5;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background: rgba(79, 70, 229, 0.05);
        margin-bottom: 25px;
    }
    
    /* Result styling */
    .result-box {
        border-radius: 10px;
        padding: 20px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 30px;
    }
    
    /* Emotion badge */
    .emotion-badge {
        font-size: 1.5rem;
        padding: 10px 20px;
        border-radius: 50px;
        display: inline-block;
        margin-top: 15px;
    }
    
    /* Animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .pulsing {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Definition ---
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



# Emotion mapping
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
EMOTION_EMOJIS = ['😐', '😌', '😄', '😔', '😠', '😨', '🤢', '😲']

# --- Feature Extraction ---
def extract_features(data, sample_rate):
    result = []
    stft_magnitude = np.abs(librosa.stft(data))
    chroma = librosa.feature.chroma_stft(S=stft_magnitude, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)
    result.extend(chroma_mean)
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    result.extend(mfccs_mean)
    zcr = librosa.feature.zero_crossing_rate(y=data)
    zcr_mean = np.mean(zcr.T, axis=0)
    result.extend(zcr_mean)
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    mel_spec_mean = np.mean(mel_spec.T, axis=0)
    result.extend(mel_spec_mean)
    rms = librosa.feature.rms(y=data)
    rms_mean = np.mean(rms.T, axis=0)
    result.extend(rms_mean)
    return np.array(result)

# --- Streamlit App ---
def main():
    # App header
    st.markdown('<h1 class="title">🎤 Speech Emotion Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar for additional info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app analyzes speech patterns to detect emotions using a deep learning model.
        - Upload a WAV audio file
        - See emotion prediction
        - Visualize audio features
        """)
        st.divider()
        st.subheader("Model Information")
        st.caption("**Architecture:** CNN-LSTM Hybrid")
        st.caption("**Accuracy:** 82.6% (Validation Set)")
        st.divider()
        st.markdown("[GitHub Repository](https://github.com/Vansh-187/speech-emotion-recognition)")
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Upload section with custom styling
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            # Audio player
            st.audio(uploaded_file)
            
            # Visualization section
            with st.spinner('Analyzing audio...'):
                # Save temporary file
                with open("temp.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load audio
                data, sr = librosa.load("temp.wav", duration=2.5, offset=0.6)
                
                # Create waveform visualization
                fig, ax = plt.subplots(figsize=(10, 3))
                librosa.display.waveshow(data, sr=sr, ax=ax)
                ax.set_title("Audio Waveform", fontsize=14)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig, use_container_width=True)  # Added use_container_width
    
    with col2:
        if uploaded_file:
            # Display analysis results
            with st.spinner('Processing emotion...'):
                # Feature extraction
                features = extract_features(data, sr)
                
                # Validate feature length
                if len(features) != 162:
                    st.error(f"Feature extraction error: Expected 162 features, got {len(features)}")
                else:
                    # Load model
                    device = torch.device('cpu')
                    model = CNNLSTM(num_classes=8).to(device)
                    model_path = os.path.join('models','best_model.pth')
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    
                    # Prepare input tensor
                    features_tensor = torch.tensor(features, dtype=torch.float32).view(1, 1, 9, 18)
                    
                    # Predict
                    with torch.no_grad():
                        output = model(features_tensor)
                        pred = torch.argmax(output, dim=1).item()
                    
                    # Add artificial delay for better UX
                    time.sleep(1.5)
                    
                # Display results with animation
                emotion = EMOTION_LABELS[pred]
                emoji = EMOTION_EMOJIS[pred]
                
                # Color coding for emotions
                emotion_colors = {
                    'neutral': '#94a3b8',
                    'calm': '#60a5fa',
                    'happy': '#fbbf24',
                    'sad': '#60a5fa',
                    'angry': '#f87171',
                    'fear': '#c084fc',
                    'disgust': '#34d399',
                    'surprise': '#f472b6'
                }
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader("Analysis Results")
                
                # Emotion badge with pulsing animation
                st.markdown(
                    f'<div class="emotion-badge pulsing" style="background-color: {emotion_colors[emotion]}; color: white;">'
                    f'{emoji} {emotion.capitalize()}'
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Confidence meter
                confidence = torch.softmax(output, dim=1)[0][pred].item() * 100
                st.progress(int(confidence), text=f"Confidence: {confidence:.1f}%")
                
                # Feature visualization
                st.divider()
                st.subheader("Audio Features")
                
                # Create feature importance chart
                feature_types = ['Chroma', 'MFCC', 'ZCR', 'Mel', 'RMS']
                feature_lengths = [12, 20, 1, 128, 1]  # Adjust to match your feature extractor
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(feature_types, feature_lengths, color='#4f46e5')
                ax.set_title("Feature Composition", fontsize=14)
                st.pyplot(fig)
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()


