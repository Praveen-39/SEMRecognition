"""
Smoke test to import the Streamlit app and run its initialization without launching the server.
It sets an environment variable SKIP_MODEL_LOAD to instruct the app to avoid loading TensorFlow models during the test.
"""
import os
os.environ['SKIP_MODEL_LOAD'] = '1'

# Import the app module
from results import ph5_streamlit

print('Imported ph5_streamlit OK')

# Try to instantiate the app class and call methods that don't load models
app = ph5_streamlit.EmotionSarcasmApp()
print('Instantiated EmotionSarcasmApp OK')

# Test feature extraction on a short silent buffer created programmatically
import numpy as np
import soundfile as sf

# create 1-second silent wav
sr = 22050
y = np.zeros(sr, dtype=np.float32)
path = 'test_silence.wav'
sf.write(path, y, sr)

features = app.extract_audio_features(path)
print('Extracted features length:', len(features))
