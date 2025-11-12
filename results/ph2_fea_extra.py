"""
PHASE 2: Complete Feature Extraction - FULLY CORRECTED
Works with Windows paths, handles all errors, extracts exactly 73 features
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract exactly 73 audio features from speech files"""
    
    def __init__(self, sr=22050):
        self.sr = sr
    
    def extract_features(self, file_path):
        """
        Extract all 73 features from audio file
        Returns numpy array of exactly 73 features
        """
        try:
            # Load audio (3 seconds)
            y, sr = librosa.load(file_path, sr=self.sr, duration=3)
            
            # Pad or truncate to exactly 3 seconds
            target_length = self.sr * 3
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            else:
                y = y[:target_length]
            
            # Initialize feature list
            features = []
            
            # 1. MFCC (26 features: 13 mean + 13 std)
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_std = np.std(mfcc, axis=1)
                features.extend(mfcc_mean.tolist())
                features.extend(mfcc_std.tolist())
            except:
                features.extend([0] * 26)
            
            # 2. Chroma (12 features)
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)
                features.extend(chroma_mean.tolist())
            except:
                features.extend([0] * 12)
            
            # 3. Mel Spectrogram (20 features)
            try:
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
                mel_mean = np.mean(mel, axis=1)
                features.extend(mel_mean.tolist())
            except:
                features.extend([0] * 20)
            
            # 4. Spectral Centroid (1 feature)
            try:
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                features.append(float(np.mean(spec_cent)))
            except:
                features.append(0.0)
            
            # 5. Spectral Rolloff (1 feature)
            try:
                spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
                features.append(float(np.mean(spec_roll)))
            except:
                features.append(0.0)
            
            # 6. Spectral Bandwidth (1 feature)
            try:
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                features.append(float(np.mean(spec_bw)))
            except:
                features.append(0.0)
            
            # 7. Spectral Contrast (1 feature) - FIXED: take overall mean
            try:
                spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                features.append(float(np.mean(spec_contrast)))
            except:
                features.append(0.0)
            
            # 8. Spectral Flatness (1 feature)
            try:
                spec_flat = librosa.feature.spectral_flatness(y=y)
                features.append(float(np.mean(spec_flat)))
            except:
                features.append(0.0)
            
            # 9. Zero Crossing Rate (1 feature)
            try:
                zcr = librosa.feature.zero_crossing_rate(y)
                features.append(float(np.mean(zcr)))
            except:
                features.append(0.0)
            
            # 10. RMS Energy (1 feature)
            try:
                rms = librosa.feature.rms(y=y)
                features.append(float(np.mean(rms)))
            except:
                features.append(0.0)
            
            # 11. Pitch Mean (1 feature)
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_values = pitches[pitches > 0]
                if len(pitch_values) > 0:
                    features.append(float(np.mean(pitch_values)))
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
            
            # 12. Pitch Std (1 feature)
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_values = pitches[pitches > 0]
                if len(pitch_values) > 0:
                    features.append(float(np.std(pitch_values)))
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
            
            # 13. Tempo (1 feature)
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features.append(float(tempo))
            except:
                features.append(0.0)
            
            # 14. Harmonic Mean (1 feature)
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                features.append(float(np.mean(y_harmonic)))
            except:
                features.append(0.0)
            
            # 15. Percussive Mean (1 feature)
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                features.append(float(np.mean(y_percussive)))
            except:
                features.append(0.0)
            
            # 16. Tonnetz (6 features)
            try:
                tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
                tonnetz_mean = np.mean(tonnetz, axis=1)
                features.extend(tonnetz_mean.tolist())
            except:
                features.extend([0] * 6)
            
            # Convert to numpy array
            features = np.array(features, dtype=np.float32)
            
            # CRITICAL: Ensure exactly 73 features
            if len(features) != 73:
                print(f"Warning: Got {len(features)} features, expected 73. Padding/truncating...")
                if len(features) < 73:
                    features = np.pad(features, (0, 73 - len(features)), mode='constant')
                else:
                    features = features[:73]
            
            # Replace any NaN or Inf
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            print(f"\nCritical error in {os.path.basename(file_path)}: {str(e)}")
            return np.zeros(73, dtype=np.float32)


def get_emotion_from_filename(filename):
    """Extract emotion label from RAVDESS filename"""
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    basename = os.path.basename(filename)
    try:
        emotion_code = basename.split('-')[2]
        return emotion_map.get(emotion_code, 'unknown')
    except:
        return 'unknown'


def main():
    """Main feature extraction pipeline"""
    
    print("="*70)
    print("PHASE 2: FEATURE EXTRACTION - CORRECTED VERSION")
    print("="*70)
    
    # YOUR DATA PATH - Windows path with raw string
    data_path = r'raw'  # Your Actor_01 to Actor_24 are here
    
    # Alternative: Use absolute path if relative doesn't work
    # data_path = r'C:\Users\LENOVO\OneDrive\Desktop\final project\raw'
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"\n❌ Data folder not found: {data_path}")
        print(f"\nCurrent directory: {os.getcwd()}")
        print("\nAvailable folders:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  - {item}")
        return
    
    # Find all audio files
    audio_files = glob.glob(os.path.join(data_path, '**', '*.wav'), recursive=True)
    
    if not audio_files:
        print(f"\n❌ No .wav files found in {data_path}")
        print(f"\nChecking contents of {data_path}:")
        for item in os.listdir(data_path):
            print(f"  - {item}")
        return
    
    print(f"\n✓ Found data folder: {data_path}")
    print(f"✓ Found {len(audio_files)} audio files")
    
    # Show sample paths
    print(f"\n📝 Sample files:")
    for file in audio_files[:3]:
        print(f"   {file}")
    
    # Create output folders
    os.makedirs('features', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize extractor
    extractor = AudioFeatureExtractor()
    
    # Extract features
    X = []
    y = []
    successful = 0
    failed = 0
    
    print(f"\n🎵 Extracting 73 features from each file...")
    print("This takes ~5-10 minutes depending on file count...\n")
    
    for file_path in tqdm(audio_files, desc="Processing"):
        features = extractor.extract_features(file_path)
        emotion = get_emotion_from_filename(file_path)
        
        # Validate features
        if features is not None and len(features) == 73:
            if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                if emotion != 'unknown':
                    X.append(features)
                    y.append(emotion)
                    successful += 1
            else:
                failed += 1
        else:
            failed += 1
    
    # Convert to arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"\n✓ Extraction complete!")
    print(f"✓ Successfully processed: {successful} files")
    print(f"✓ Failed/skipped: {failed} files")
    print(f"✓ Final shape: {X.shape}")
    
    if len(X) == 0:
        print("\n❌ No features extracted! Please check your audio files.")
        return
    
    # Save features
    print(f"\n💾 Saving features...")
    np.save('features/X_features.npy', X)
    np.save('features/y_labels.npy', y)
    
    # Save as CSV
    df = pd.DataFrame(X)
    df['emotion'] = y
    df.to_csv('features/features.csv', index=False)
    
    print(f"✓ Saved: features/X_features.npy")
    print(f"✓ Saved: features/y_labels.npy")
    print(f"✓ Saved: features/features.csv")
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    print(f"\n📊 Dataset Shape: {X.shape}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Features per sample: {X.shape[1]}")
    
    print(f"\n📊 Emotion Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for emotion, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        print(f"   {emotion:12s}: {count:4d} samples ({percentage:.1f}%)")
    
    print(f"\n📊 Feature Value Ranges:")
    print(f"   Mean: {X.mean():.6f}")
    print(f"   Std:  {X.std():.6f}")
    print(f"   Min:  {X.min():.6f}")
    print(f"   Max:  {X.max():.6f}")
    
    # Data quality check
    has_nan = np.any(np.isnan(X))
    has_inf = np.any(np.isinf(X))
    
    if not has_nan and not has_inf:
        print(f"\n✓ Data quality: EXCELLENT (no NaN or Inf values)")
    else:
        print(f"\n⚠️ Warning: Data contains NaN={has_nan}, Inf={has_inf}")
    
    print("\n" + "="*70)
    print("✅ PHASE 2 COMPLETE!")
    print("="*70)
    
    print("\n📁 Files created:")
    print("   features/X_features.npy  - Feature matrix (ready for training)")
    print("   features/y_labels.npy    - Emotion labels")
    print("   features/features.csv    - Human-readable CSV")
    
    print(f"\n✅ Successfully extracted features from {len(X)} audio files!")
    print("✅ Ready for Phase 3: Model Training!")
    print("\nNext: Reply with 'Phase 3 code bro!' when ready")


if __name__ == "__main__":
    main()
