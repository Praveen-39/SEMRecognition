"""
PHASE 5: Streamlit Web Application
Complete demo app for Speech Emotion Recognition with Sarcasm Detection
"""

import streamlit as st
import os
import numpy as np
import librosa
import pickle
import tensorflow as tf
from tensorflow import keras
import speech_recognition as sr
from textblob import TextBlob
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Speech Emotion & Sarcasm Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .emotion-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1f77b4;
    }
    .sarcasm-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


class EmotionSarcasmApp:
    """Main application class"""
    
    def __init__(self, model_path='models'):
        """Initialize the application"""
        self.model_path = model_path
        self.load_models()
    
    @st.cache_resource
    def load_models(_self):
        """Load trained models and artifacts"""
        # Allow skipping heavy model loading for quick smoke tests
        if os.environ.get('SKIP_MODEL_LOAD') == '1':
            # Return lightweight dummies that satisfy attribute access used elsewhere
            class DummyModel:
                def predict(self, x, verbose=0):
                    # return uniform probabilities for 8 classes (sum to 1)
                    import numpy as _np
                    batch = x.shape[0]
                    probs = _np.ones((batch, 8)) / 8.0
                    return probs

            class DummyScaler:
                def transform(self, x):
                    return x

            try:
                with open(f'{_self.model_path}/label_encoder.pkl', 'rb') as f:
                    label_encoder = pickle.load(f)
            except Exception:
                # fallback dummy label encoder
                class DummyLE:
                    classes_ = ['happy','sad','angry','fearful','disgust','surprised','neutral','calm']
                label_encoder = DummyLE()

            return DummyModel(), label_encoder, DummyScaler()

        try:
            # Load emotion model (cleaned .h5 should work)
            model = keras.models.load_model(f'{_self.model_path}/emotion_recognition_model.h5')

            # Load label encoder
            with open(f'{_self.model_path}/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)

            # Load scaler
            with open(f'{_self.model_path}/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

            return model, label_encoder, scaler

        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.info("Please ensure Phase 3 is completed and models are saved.")
            return None, None, None
    
    def extract_audio_features(self, audio_path):
        """Extract 73 features from audio file"""
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=3)
            
            target_length = 22050 * 3
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            else:
                y = y[:target_length]
            
            features = []
            
            # MFCC (26)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfcc, axis=1).tolist())
            features.extend(np.std(mfcc, axis=1).tolist())
            
            # Chroma (12)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1).tolist())
            
            # Mel (20)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
            features.extend(np.mean(mel, axis=1).tolist())
            
            # Spectral (5)
            features.append(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))))
            features.append(float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))))
            features.append(float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))))
            features.append(float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))))
            features.append(float(np.mean(librosa.feature.spectral_flatness(y=y))))
            
            # ZCR (1), RMS (1)
            features.append(float(np.mean(librosa.feature.zero_crossing_rate(y))))
            features.append(float(np.mean(librosa.feature.rms(y=y))))
            
            # Pitch (2)
            pitches, _ = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[pitches > 0]
            if len(pitch_values) > 0:
                features.append(float(np.mean(pitch_values)))
                features.append(float(np.std(pitch_values)))
            else:
                features.extend([0.0, 0.0])
            
            # Tempo (1)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(float(tempo))
            
            # Harmonic/Percussive (2)
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            features.append(float(np.mean(y_harmonic)))
            features.append(float(np.mean(y_percussive)))
            
            # Tonnetz (6)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features.extend(np.mean(tonnetz, axis=1).tolist())
            
            features = np.array(features, dtype=np.float32)
            
            if len(features) != 73:
                if len(features) < 73:
                    features = np.pad(features, (0, 73 - len(features)), mode='constant')
                else:
                    features = features[:73]
            
            return features
            
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return np.zeros(73, dtype=np.float32)
    
    def predict_emotion(self, audio_path):
        """Predict emotion from audio"""
        model, label_encoder, scaler = self.load_models()
        
        if model is None:
            return None, None, None
        
        # Extract features
        features = self.extract_audio_features(audio_path)
        
        # Scale
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Reshape
        features_reshaped = features_scaled.reshape(1, 73, 1)
        
        # Predict
        predictions = model.predict(features_reshaped, verbose=0)
        
        emotion_idx = np.argmax(predictions[0])
        emotion = label_encoder.classes_[emotion_idx]
        confidence = predictions[0][emotion_idx]
        
        return emotion, confidence, predictions[0]
    
    def speech_to_text(self, audio_path):
        """Convert speech to text"""
        recognizer = sr.Recognizer()
        
        try:
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            
            text = recognizer.recognize_google(audio)
            return text
            
        except:
            return None
    
    def analyze_sentiment(self, text):
        """Analyze text sentiment"""
        if not text:
            return 'neutral', 0.0, 0.0
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return sentiment, polarity, subjectivity
    
    def detect_sarcasm(self, emotion, text_sentiment, confidence):
        """Detect sarcasm"""
        positive_emotions = ['happy', 'surprised', 'calm']
        negative_emotions = ['angry', 'sad', 'fearful', 'disgust']
        
        emotion_positive = emotion in positive_emotions
        emotion_negative = emotion in negative_emotions
        sentiment_positive = text_sentiment == 'positive'
        sentiment_negative = text_sentiment == 'negative'
        
        sarcasm_score = 0.0
        indicators = []
        
        if emotion_negative and sentiment_positive:
            sarcasm_score += 0.6
            indicators.append("Negative emotion with positive words")
        
        if emotion_positive and sentiment_negative:
            sarcasm_score += 0.5
            indicators.append("Positive emotion with negative words")
        
        if confidence > 0.8 and sarcasm_score > 0:
            sarcasm_score += 0.2
            indicators.append("High emotion confidence")
        
        sarcasm_detected = sarcasm_score >= 0.5
        
        return sarcasm_detected, sarcasm_score, indicators


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🎭 Speech Emotion & Sarcasm Detector</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Emotion Recognition with Sarcasm Detection</div>', 
                unsafe_allow_html=True)
    
    # Initialize app
    app = EmotionSarcasmApp()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.info("""
        **Features:**
        - Emotion Recognition (8 emotions)
        - Speech-to-Text Conversion
        - Sentiment Analysis
        - Sarcasm Detection
        
        **How it works:**
        1. Upload or record audio
        2. AI analyzes tone and content
        3. Detects emotion and sarcasm
        """)
        
        st.header("🎯 Emotions Detected")
        emotions = ['😊 Happy', '😢 Sad', '😠 Angry', '😨 Fearful', 
                   '🤢 Disgust', '😮 Surprised', '😐 Neutral', '😌 Calm']
        for emotion in emotions:
            st.write(emotion)
    
    # Main content
    st.header("📤 Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=['wav'],
        help="Upload a WAV audio file (max 3 seconds will be analyzed)"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Analyze button
        if st.button("🔍 Analyze Audio", type="primary"):
            with st.spinner("Analyzing audio... Please wait..."):
                # 1. Predict emotion
                emotion, confidence, all_predictions = app.predict_emotion(tmp_path)
                
                if emotion is not None:
                    # 2. Speech to text
                    text = app.speech_to_text(tmp_path)
                    
                    # 3. Sentiment analysis
                    sentiment = 'unknown'
                    polarity = 0.0
                    subjectivity = 0.0
                    
                    if text:
                        sentiment, polarity, subjectivity = app.analyze_sentiment(text)
                    
                    # 4. Sarcasm detection
                    sarcasm_detected = False
                    sarcasm_score = 0.0
                    indicators = []
                    
                    if text:
                        sarcasm_detected, sarcasm_score, indicators = app.detect_sarcasm(
                            emotion, sentiment, confidence
                        )
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Create columns
                    col1, col2 = st.columns(2)
                    
                    # Emotion results
                    with col1:
                        st.markdown('<div class="result-box emotion-box">', unsafe_allow_html=True)
                        st.subheader("😊 Emotion Detection")
                        
                        emotion_emoji = {
                            'happy': '😊', 'sad': '😢', 'angry': '😠',
                            'fearful': '😨', 'disgust': '🤢', 'surprised': '😮',
                            'neutral': '😐', 'calm': '😌'
                        }
                        
                        st.metric(
                            "Detected Emotion",
                            f"{emotion_emoji.get(emotion, '🎭')} {emotion.upper()}",
                            f"{confidence*100:.1f}% confidence"
                        )
                        
                        # Emotion distribution chart
                        _, label_encoder, _ = app.load_models()
                        emotion_df = pd.DataFrame({
                            'Emotion': label_encoder.classes_,
                            'Probability': all_predictions * 100
                        })
                        emotion_df = emotion_df.sort_values('Probability', ascending=True)
                        
                        fig = px.bar(emotion_df, x='Probability', y='Emotion',
                                    orientation='h',
                                    color='Probability',
                                    color_continuous_scale='Blues')
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Sarcasm results
                    with col2:
                        st.markdown('<div class="result-box sarcasm-box">', unsafe_allow_html=True)
                        st.subheader("🎭 Sarcasm Detection")
                        
                        if text:
                            if sarcasm_detected:
                                st.error(f"⚠️ SARCASM DETECTED!")
                                st.metric("Sarcasm Score", f"{sarcasm_score:.2f}", "High")
                            else:
                                st.success("✅ No Sarcasm Detected")
                                st.metric("Sarcasm Score", f"{sarcasm_score:.2f}", "Low")
                            
                            # Sarcasm gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=sarcasm_score * 100,
                                title={'text': "Sarcasm Probability"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "orange" if sarcasm_detected else "green"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgreen"},
                                        {'range': [50, 100], 'color': "lightyellow"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 50
                                    }
                                }
                            ))
                            fig.update_layout(height=250)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            if indicators:
                                st.write("**Indicators:**")
                                for indicator in indicators:
                                    st.write(f"• {indicator}")
                        else:
                            st.warning("⚠️ Could not transcribe audio")
                            st.info("Sarcasm detection requires text transcription")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Text analysis
                    if text:
                        st.subheader("📝 Transcription & Sentiment")
                        
                        col3, col4 = st.columns([2, 1])
                        
                        with col3:
                            st.text_area("Transcribed Text", text, height=100)
                        
                        with col4:
                            st.metric("Sentiment", sentiment.upper())
                            st.metric("Polarity", f"{polarity:.3f}")
                            st.metric("Subjectivity", f"{subjectivity:.3f}")
                    
                    # Clean up
                    os.unlink(tmp_path)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎯 Speech Emotion Recognition with Sarcasm Detection</p>
        <p>Powered by Deep Learning | CNN-LSTM Architecture</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
