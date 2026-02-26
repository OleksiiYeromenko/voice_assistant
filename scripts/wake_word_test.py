import pyaudio
import numpy as np
from openwakeword.model import Model
import time

# ==========================================
# Configuration
# ==========================================
# Update this to match your trained model's name and path
MODEL_PATH = "./my_custom_model/hey_orioon.onnx" 

# PyAudio configuration for openWakeWord
CHUNK = 1280            # 1280 samples per chunk
FORMAT = pyaudio.paInt16 # 16-bit audio
CHANNELS = 1            # Mono
RATE = 16000            # 16 kHz sample rate

# ==========================================
# Initialization
# ==========================================
print(f"Loading custom openWakeWord model from: {MODEL_PATH}")
# Initialize the openwakeword model
owwModel = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")

# Get the internal name the model uses (usually the filename without extension)
model_name = list(owwModel.models.keys())[0]

audio = pyaudio.PyAudio()

# Open the microphone stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("\nListening for your wake word... (Press Ctrl+C to stop)")
print("-" * 50)

# ==========================================
# Live Inference Loop
# ==========================================
try:
    while True:
        # Read a chunk of audio from the microphone
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # Convert the byte data to a numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Feed the audio data into the model
        prediction = owwModel.predict(audio_data)
        
        # Extract the confidence score (0.0 to 1.0)
        score = prediction[model_name]
        
        # If the score is above a certain threshold, we consider it a detection
        # You can lower this to 0.4 if it's not catching it, or raise to 0.7 if it triggers too easily
        if score > 0.5:
            print(f"[{time.strftime('%H:%M:%S')}] Wake word detected! (Confidence Score: {score:.3f})")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    # Clean up the audio stream gracefully
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("Microphone stream closed.")