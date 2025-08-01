from flask import Flask, request, jsonify
import whisper
from transformers import pipeline
import numpy as np
import tempfile
import soundfile as sf
import os
import base64
import io
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables to store loaded models and latest transcript
whisper_model = None
emotion_pipeline = None
latest_transcript = None

def load_models():
    """Load models once when the server starts"""
    global whisper_model, emotion_pipeline
    try:
        logger.info("Loading Whisper model...")
        whisper_model = whisper.load_model("/Users/mac/Desktop/hackerthon/Facial Emotion/models/whisper_model/base.pt")
        logger.info("Whisper model loaded successfully")

        logger.info("Loading Emotion Recognition model...")
        emotion_pipeline = pipeline(
            "text-classification",
            model="/Users/mac/Desktop/hackerthon/Facial Emotion/emotion_english_distilroberta_model",
            top_k=None
        )
        logger.info("Emotion model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

def transcribe_audio_file(audio_path):
    """Transcribe audio file using Whisper"""
    try:
        result = whisper_model.transcribe(audio_path)
        text = result.get("text", "").strip()
        return text
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return None

def analyze_emotions_from_text(text):
    """Analyze emotions from text"""
    if not text:
        return []
    try:
        results = emotion_pipeline(text)
        sorted_emotions = sorted(results[0], key=lambda x: x["score"], reverse=True)
        emotions = []
        for emotion in sorted_emotions:
            emotions.append({
                "label": emotion["label"],
                "score": round(emotion["score"] * 100, 2)
            })
        return emotions
    except Exception as e:
        logger.error(f"Error analyzing emotions: {e}")
        return []

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Speech Emotion Analysis API is running",
        "models_loaded": {
            "whisper": whisper_model is not None,
            "emotion": emotion_pipeline is not None
        }
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    """Transcribe audio file to text"""
    try:
        if whisper_model is None:
            return jsonify({"error": "Whisper model not loaded"}), 500
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            audio_file.save(tmp_file.name)
            text = transcribe_audio_file(tmp_file.name)
            os.unlink(tmp_file.name)

        if text is None:
            return jsonify({"error": "Failed to transcribe audio"}), 500

        global latest_transcript
        latest_transcript = text

        return jsonify({
            "success": True,
            "transcribed_text": text,
            "message": "Audio transcribed successfully" if text else "No speech detected"
        })

    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_emotions', methods=['POST'])
def analyze_emotions_endpoint():
    """Analyze emotions from text"""
    try:
        if emotion_pipeline is None:
            return jsonify({"error": "Emotion model not loaded"}), 500
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        emotions = analyze_emotions_from_text(text)
        if not emotions:
            return jsonify({"error": "Failed to analyze emotions"}), 500
        return jsonify({
            "success": True,
            "text": text,
            "emotions": emotions,
            "dominant_emotion": emotions[0] if emotions else None
        })
    except Exception as e:
        logger.error(f"Error in analyze_emotions endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe_and_analyze', methods=['POST'])
def transcribe_and_analyze_endpoint():
    """Combined endpoint: transcribe audio and analyze emotions"""
    try:
        if whisper_model is None or emotion_pipeline is None:
            return jsonify({"error": "Models not loaded"}), 500
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            audio_file.save(tmp_file.name)
            text = transcribe_audio_file(tmp_file.name)
            os.unlink(tmp_file.name)

        if text is None:
            return jsonify({"error": "Failed to transcribe audio"}), 500

        global latest_transcript
        latest_transcript = text

        emotions = []
        if text:
            emotions = analyze_emotions_from_text(text)

        return jsonify({
            "success": True,
            "transcribed_text": text,
            "emotions": emotions,
            "dominant_emotion": emotions[0] if emotions else None,
            "message": "Audio processed successfully" if text else "No speech detected"
        })

    except Exception as e:
        logger.error(f"Error in transcribe_and_analyze endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe_base64', methods=['POST'])
def transcribe_base64_endpoint():
    """Transcribe audio from base64 encoded data"""
    try:
        if whisper_model is None:
            return jsonify({"error": "Whisper model not loaded"}), 500
        data = request.get_json()
        if not data or 'audio_data' not in data:
            return jsonify({"error": "No audio data provided"}), 400
        try:
            audio_bytes = base64.b64decode(data['audio_data'])
        except Exception:
            return jsonify({"error": "Invalid base64 audio data"}), 400

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file.flush()
            text = transcribe_audio_file(tmp_file.name)
            os.unlink(tmp_file.name)

        if text is None:
            return jsonify({"error": "Failed to transcribe audio"}), 500

        global latest_transcript
        latest_transcript = text

        return jsonify({
            "success": True,
            "transcribed_text": text,
            "message": "Audio transcribed successfully" if text else "No speech detected"
        })

    except Exception as e:
        logger.error(f"Error in transcribe_base64 endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_latest_transcript', methods=['GET'])
def get_latest_transcript():
    """Get the latest transcribed text"""
    try:
        if latest_transcript is None:
            return jsonify({"error": "No transcript available"}), 404
        return jsonify({
            "success": True,
            "transcribed_text": latest_transcript,
            "message": "Latest transcript retrieved"
        })
    except Exception as e:
        logger.error(f"Error in get_latest_transcript endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    try:
        load_models()
        app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
        print("Starting Speech Emotion Analysis API...")
        print("Available endpoints:")
        print("  GET  /health - Health check")
        print("  POST /transcribe - Transcribe audio file")
        print("  POST /analyze_emotions - Analyze emotions from text")
        print("  POST /transcribe_and_analyze - Combined transcription and emotion analysis")
        print("  POST /transcribe_base64 - Transcribe from base64 audio data")
        print("  GET  /get_latest_transcript - Get latest transcribed text")
        print("\nServer starting on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"Error: {e}")
