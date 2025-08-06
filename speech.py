import whisper
from transformers import pipeline
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf

def record_audio(duration=5, fs=16000):
    print(f"Recording audio for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    print("Recording complete.")
    return audio, fs

def save_audio_to_wav(audio, fs, filename):
    sf.write(filename, audio, fs)

def transcribe_audio(model, audio_path):
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    text = result.get("text", "").strip()
    if not text:
        print("No speech detected.")
    else:
        print(f"Transcribed Text: {text}")
    return text

def analyze_emotions(emotion_pipeline, text):
    if not text:
        return
    print("Analyzing emotions...")
    results = emotion_pipeline(text)
    sorted_emotions = sorted(results[0], key=lambda x: x["score"], reverse=True)
    print("Emotion Results:")
    for emotion in sorted_emotions:
        print(f"  {emotion['label']}: {emotion['score'] * 100:.2f}%")
    dominant = sorted_emotions[0]
    print(f"Dominant Emotion: {dominant['label']} ({dominant['score']*100:.2f}%)")

def main():
    # Load models once
    print("Loading Whisper model (this may take a while)...")
    whisper_model = whisper.load_model("/Users/mac/Desktop/hackerthon/Facial Emotion/models/whisper_model/base.pt")
    print("Loading Emotion Recognition model...")
    emotion_pipeline = pipeline(
        "text-classification",
        model="/Users/mac/Desktop/hackerthon/Facial Emotion/emotion_english_distilroberta_model",
        top_k=None
    )
    
    while True:
        try:
            audio, fs = record_audio(duration=5)
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
                save_audio_to_wav(audio, fs, tmpfile.name)
                text = transcribe_audio(whisper_model, tmpfile.name)
                analyze_emotions(emotion_pipeline, text)
            print("\n--- Ready for next recording ---\n")
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
