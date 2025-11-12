"""
Data Exploration Script
Analyzes RAVDESS dataset and generates visualizations
"""

import os
import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def get_emotion_from_filename(filename):
    """Extract emotion from RAVDESS filename"""
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

def explore_dataset(data_path="D://Speech Project//raw"):
    """Explore and visualize dataset"""
    
    print("="*70)
    print("RAVDESS Dataset Exploration")
    print("="*70)
    
    # Find audio files
    audio_files = glob.glob(f'{data_path}/**/*.wav', recursive=True)
    
    if not audio_files:
        print(f"\n❌ No audio files found in {data_path}")
        print("Please ensure RAVDESS dataset is properly installed.")
        return
    
    print(f"\n✓ Total audio files found: {len(audio_files)}")
    
    # Extract metadata
    data = []
    print("\nAnalyzing files...")
    for filepath in audio_files[:10]:  # Show first 10
        filename = os.path.basename(filepath)
        print(f"  • {filename} -> {get_emotion_from_filename(filepath)}")
    
    for filepath in audio_files:
        filename = os.path.basename(filepath)
        parts = filename.replace('.wav', '').split('-')
        
        if len(parts) == 7:
            data.append({
                'filepath': filepath,
                'filename': filename,
                'emotion': get_emotion_from_filename(filepath),
                'actor': parts[6],
                'intensity': 'normal' if parts[3] == '01' else 'strong'
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Gender (odd actors = male, even = female)
    df['gender'] = df['actor'].apply(
        lambda x: 'male' if int(x) % 2 == 1 else 'female'
    )
    
    # Statistics
    print("\n" + "="*70)
    print("Dataset Statistics")
    print("="*70)
    print(f"\nTotal samples: {len(df)}")
    print(f"Unique emotions: {df['emotion'].nunique()}")
    print(f"Unique actors: {df['actor'].nunique()}")
    
    print("\n📊 Emotion Distribution:")
    print(df['emotion'].value_counts().sort_index())
    
    print("\n👥 Gender Distribution:")
    print(df['gender'].value_counts())
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # 1. Emotion distribution plot
    print("\n📊 Creating emotion distribution plot...")
    plt.figure(figsize=(12, 6))
    emotion_counts = df['emotion'].value_counts().sort_index()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
              '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
    emotion_counts.plot(kind='bar', color=colors, edgecolor='black', linewidth=1.2)
    plt.title('Emotion Distribution in RAVDESS Dataset', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/emotion_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/emotion_distribution.png")
    plt.show()
    
    # 2. Sample waveforms for each emotion
    print("\n🎵 Creating sample waveforms for each emotion...")
    sample_emotions = df.groupby('emotion').first()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (emotion, row) in enumerate(sample_emotions.iterrows()):
        try:
            y, sr = librosa.load(row['filepath'], sr=22050, duration=3)
            
            axes[idx].plot(np.linspace(0, len(y)/sr, len(y)), y, 
                          color=colors[idx], linewidth=0.8)
            axes[idx].set_title(emotion.upper(), fontweight='bold', fontsize=11)
            axes[idx].set_xlabel('Time (s)', fontsize=9)
            axes[idx].set_ylabel('Amplitude', fontsize=9)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim(-1, 1)
        except Exception as e:
            print(f"⚠️ Error loading {emotion}: {e}")
    
    plt.suptitle('Sample Waveforms by Emotion', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results/sample_waveforms.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/sample_waveforms.png")
    plt.show()
    
    # 3. Detailed audio analysis
    print("\n🎵 Creating detailed audio analysis...")
    sample_file = df[df['emotion'] == 'happy'].iloc[0]['filepath']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Load audio
    y, sr = librosa.load(sample_file, sr=22050)
    
    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#4ECDC4')
    axes[0].set_title('Waveform (Happy Emotion Sample)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                                   ax=axes[1], cmap='viridis')
    axes[1].set_title('Spectrogram', fontweight='bold', fontsize=12)
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[2], cmap='coolwarm')
    axes[2].set_title('MFCC (Mel-Frequency Cepstral Coefficients)', 
                     fontweight='bold', fontsize=12)
    axes[2].set_ylabel('MFCC Coefficients')
    fig.colorbar(img, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('results/detailed_audio_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/detailed_audio_analysis.png")
    plt.show()
    
    # Save metadata
    df.to_csv('results/dataset_metadata.csv', index=False)
    print("\n✓ Saved: results/dataset_metadata.csv")
    
    # Summary
    print("\n" + "="*70)
    print("✅ Data Exploration Complete!")
    print("="*70)
    print(f"\n📁 Results saved in: results/")
    print("   • emotion_distribution.png")
    print("   • sample_waveforms.png")
    print("   • detailed_audio_analysis.png")
    print("   • dataset_metadata.csv")
    
    # Check balance
    if emotion_counts.std() < 10:
        print("\n✓ Dataset is well-balanced across emotions")
    else:
        print("\n⚠️ Dataset shows some imbalance")
    
    print("\n" + "="*70)
    print("Dataset Summary:")
    print("="*70)
    print(f"Total Samples: {len(df)}")
    print(f"Emotions: {', '.join(df['emotion'].unique())}")
    print(f"Actors: {df['actor'].nunique()} (12 male, 12 female)")
    print(f"Average samples per emotion: {len(df) // df['emotion'].nunique()}")
    
    return df

if __name__ == "__main__":
    df = explore_dataset()
    
    if df is not None:
        print("\n" + "="*70)
        print("🎉 PHASE 1 COMPLETE!")
        print("="*70)
        print("\n✅ Your dataset is ready for feature extraction!")
        print("\nNext: Run Phase 2 - Feature Extraction")
