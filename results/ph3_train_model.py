"""
PHASE 3: Emotion Recognition Model Training
Train CNN-LSTM model on extracted features
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')


def load_features(features_path='features'):
    """Load extracted features and labels"""
    print("="*70)
    print("PHASE 3: MODEL TRAINING")
    print("="*70)
    
    print("\n📁 Loading features...")
    
    # Load features and labels
    X = np.load(f'{features_path}/X_features.npy')
    y = np.load(f'{features_path}/y_labels.npy')
    
    print(f"✓ Loaded features: {X.shape}")
    print(f"✓ Loaded labels: {y.shape}")
    
    return X, y


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """Preprocess features and labels for training"""
    print("\n🔧 Preprocessing data...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"✓ Label encoding complete")
    print(f"  Classes: {label_encoder.classes_}")
    print(f"  Number of classes: {len(label_encoder.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    print(f"\n✓ Data split complete")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Feature scaling complete")
    
    # Reshape for CNN-LSTM (samples, timesteps, features)
    # We treat each feature as a timestep
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_test_cat = to_categorical(y_test, num_classes=len(label_encoder.classes_))
    
    print(f"✓ Data reshaping complete")
    print(f"  X_train shape: {X_train_reshaped.shape}")
    print(f"  X_test shape: {X_test_reshaped.shape}")
    
    return X_train_reshaped, X_test_reshaped, y_train_cat, y_test_cat, y_train, y_test, label_encoder, scaler


def build_cnn_lstm_model(input_shape, num_classes):
    """Build CNN-LSTM hybrid model"""
    print("\n🏗️ Building CNN-LSTM model...")
    
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(128, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(256, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM layers for temporal patterns
        LSTM(128, return_sequences=True),
        Dropout(0.4),
        
        LSTM(64),
        Dropout(0.4),
        
        # Dense layers for classification
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model built successfully")
    print(f"\n📊 Model Summary:")
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train the model with callbacks"""
    print("\n🎯 Training model...")
    
    # Create model directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'models/best_emotion_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    callbacks = [early_stopping, model_checkpoint, reduce_lr]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Training complete!")
    
    return history


def evaluate_model(model, X_test, y_test, y_test_original, label_encoder):
    """Evaluate model performance"""
    print("\n📊 Evaluating model...")
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(y_test_original, y_pred)
    print(f"\n✓ Test Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(
        y_test_original, y_pred,
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_original, y_pred)
    
    return accuracy, cm, y_pred


def plot_training_history(history, output_path='results'):
    """Plot training history"""
    print("\n📊 Creating training visualizations...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/training_history.png")
    plt.close()


def plot_confusion_matrix(cm, label_encoder, output_path='results'):
    """Plot confusion matrix"""
    print("📊 Creating confusion matrix...")
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontweight='bold', fontsize=14)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/confusion_matrix.png")
    plt.close()


def save_model_artifacts(model, label_encoder, scaler, output_path='models'):
    """Save all model artifacts"""
    print("\n💾 Saving model artifacts...")
    
    # Save final model
    model.save(f'{output_path}/emotion_recognition_model.h5')
    print(f"✓ Saved: {output_path}/emotion_recognition_model.h5")
    
    # Save label encoder
    import pickle
    with open(f'{output_path}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Saved: {output_path}/label_encoder.pkl")
    
    # Save scaler
    with open(f'{output_path}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved: {output_path}/scaler.pkl")


def main():
    """Main training pipeline"""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 1. Load features
    X, y = load_features()
    
    # 2. Preprocess data
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig, label_encoder, scaler = preprocess_data(X, y)
    
    # 3. Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_encoder.classes_)
    model = build_cnn_lstm_model(input_shape, num_classes)
    
    # 4. Train model
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # 5. Evaluate model
    accuracy, cm, y_pred = evaluate_model(model, X_test, y_test, y_test_orig, label_encoder)
    
    # 6. Create visualizations
    plot_training_history(history)
    plot_confusion_matrix(cm, label_encoder)
    
    # 7. Save model artifacts
    save_model_artifacts(model, label_encoder, scaler)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ PHASE 3 COMPLETE!")
    print("="*70)
    
    print(f"\n🎯 Final Results:")
    print(f"   Test Accuracy: {accuracy*100:.2f}%")
    print(f"   Model saved: models/emotion_recognition_model.h5")
    print(f"   Best model: models/best_emotion_model.h5")
    
    print(f"\n📁 Generated Files:")
    print("   models/emotion_recognition_model.h5  - Final trained model")
    print("   models/best_emotion_model.h5         - Best model (highest val accuracy)")
    print("   models/label_encoder.pkl             - Label encoder")
    print("   models/scaler.pkl                    - Feature scaler")
    print("   results/training_history.png         - Training curves")
    print("   results/confusion_matrix.png         - Confusion matrix")
    
    print("\n✅ Model training successful!")
    print("✅ Ready for Phase 4: Sarcasm Detection Module!")


if __name__ == "__main__":
    main()
