from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from keras.models import load_model
import random
import time
import base64
import io
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
face_emotion_model = None
face_detector = None

# Emotion labels and responses (from your original code)
labels_dict = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 
    3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

responses = {
    "Angry": ["Whoa! Summoning a dragon?", "Lotus tea break?"],
    "Disgust": ["Shrimp paste face!", "First time with durian?"],
    "Fear": ["Saw a village ghost?", "Don't worry, I got you!"],
    "Happy": ["Tết vibes!", "Lucky money smile!"],
    "Neutral": ["Lotus pond calm", "Waiting backstage"],
    "Sad": ["Rainy Hanoi feels", "I'll dance for you"],
    "Surprise": ["Water puppet shock!", "Free rice cake?!"]
}

# State management for consistent responses
session_state = {
    "last_update_time": 0,
    "current_response": "",
    "last_emotion": ""
}

def load_face_models():
    """Load face emotion models"""
    global face_emotion_model, face_detector
    
    try:
        logger.info("Loading Face Emotion Model...")
        face_emotion_model = load_model('model_file_30epochs.h5')
        logger.info("Face emotion model loaded successfully")
        
        logger.info("Loading Face Detector...")
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        logger.info("Face detector loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error loading face models: {e}")
        return False

def process_face_emotion(image_array, use_session_state=False):
    """Process face emotion detection from image array"""
    try:
        if face_emotion_model is None or face_detector is None:
            return {"error": "Models not loaded"}
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        
        # Detect faces
        faces = face_detector.detectMultiScale(gray, 1.3, 3)
        
        if len(faces) == 0:
            return {
                "success": True,
                "faces_detected": 0,
                "faces": [],
                "message": "No faces detected"
            }
        
        results = []
        current_time = time.time()
        
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            sub_face_img = gray[y:y+h, x:x+w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))
            
            # Predict emotion
            result = face_emotion_model.predict(reshaped, verbose=0)
            label = np.argmax(result, axis=1)[0]
            emotion = labels_dict[label]
            confidence = float(np.max(result)) * 100
            
            # Handle response logic
            if use_session_state:
                # Use session state for consistent responses (like original webcam code)
                should_update = (emotion != session_state["last_emotion"]) or \
                              (current_time - session_state["last_update_time"] > 30)
                
                if should_update:
                    session_state["current_response"] = random.choice(
                        responses.get(emotion, ["I'm not sure how you're feeling 🤖"])
                    )
                    session_state["last_update_time"] = current_time
                    session_state["last_emotion"] = emotion
                
                response_text = session_state["current_response"]
            else:
                # Generate new response each time (for single image analysis)
                response_text = random.choice(
                    responses.get(emotion, ["I'm not sure how you're feeling 🤖"])
                )
            
            # Get all emotion scores
            emotion_scores = {}
            for i, score in enumerate(result[0]):
                emotion_scores[labels_dict[i]] = round(float(score) * 100, 2)
            
            face_result = {
                "bbox": {
                    "x": int(x), "y": int(y), 
                    "width": int(w), "height": int(h)
                },
                "emotion": emotion,
                "confidence": round(confidence, 2),
                "response": response_text,
                "emotion_scores": emotion_scores
            }
            results.append(face_result)
        
        # Find primary face (largest)
        primary_face = max(results, key=lambda f: f['bbox']['width'] * f['bbox']['height'])
        
        return {
            "success": True,
            "faces_detected": len(results),
            "primary_face": primary_face,
            "all_faces": results,
            "message": f"Detected {len(results)} face(s)"
        }
        
    except Exception as e:
        logger.error(f"Error processing face emotion: {e}")
        return {"error": str(e)}

def decode_base64_image(base64_string):
    """Decode base64 string to image array"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to PIL Image then to numpy array
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL to OpenCV format
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        
        image_array = np.array(pil_image)
        # Convert RGB to BGR for OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Face Emotion Recognition API is running",
        "models_loaded": {
            "face_emotion": face_emotion_model is not None,
            "face_detector": face_detector is not None
        }
    })

@app.route('/detect_face_emotion', methods=['POST'])
def detect_face_emotion():
    """Detect emotion from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read image
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Process emotion
        result = process_face_emotion(image, use_session_state=False)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in detect_face_emotion: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/detect_face_emotion_base64', methods=['POST'])
def detect_face_emotion_base64():
    """Detect emotion from base64 image"""
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode image
        image = decode_base64_image(data['image_data'])
        if image is None:
            return jsonify({"error": "Invalid base64 image data"}), 400
        
        # Process emotion
        result = process_face_emotion(image, use_session_state=False)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in detect_face_emotion_base64: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/webcam_emotion', methods=['POST'])
def webcam_emotion():
    """Real-time webcam emotion detection (with session state)"""
    try:
        data = request.get_json()
        if not data or 'frame_data' not in data:
            return jsonify({"error": "No frame data provided"}), 400
        
        # Decode frame
        frame = decode_base64_image(data['frame_data'])
        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400
        
        # Process with session state for consistent responses
        result = process_face_emotion(frame, use_session_state=True)
        
        if "error" in result:
            return jsonify(result), 500
        
        # Add session info for webcam mode
        result["session_info"] = {
            "last_emotion": session_state["last_emotion"],
            "response_age": time.time() - session_state["last_update_time"]
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in webcam_emotion: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset session state for webcam mode"""
    global session_state
    session_state = {
        "last_update_time": 0,
        "current_response": "",
        "last_emotion": ""
    }
    return jsonify({
        "success": True,
        "message": "Session state reset"
    })

@app.route('/get_emotion_info', methods=['GET'])
def get_emotion_info():
    """Get information about available emotions and responses"""
    return jsonify({
        "emotions": list(labels_dict.values()),
        "emotion_labels": labels_dict,
        "sample_responses": {
            emotion: responses[emotion][:1]  # Show first response only
            for emotion in responses.keys()
        }
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    try:
        # Load models
        if not load_face_models():
            print("Failed to load models. Please check model files exist:")
            print("- model_file_30epochs.h5")
            print("- haarcascade_frontalface_default.xml")
            exit(1)
        
        # Configure upload limits
        app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max
        
        print("Starting Lightweight Face Emotion Recognition API...")
        print("Available endpoints:")
        print("  GET  /health - Health check")
        print("  POST /detect_face_emotion - Detect from image file")
        print("  POST /detect_face_emotion_base64 - Detect from base64 image")
        print("  POST /webcam_emotion - Real-time webcam detection")
        print("  POST /reset_session - Reset webcam session")
        print("  GET  /get_emotion_info - Get emotion information")
        print("\nServer starting on http://localhost:5001")
        
        # Run on different port to avoid conflicts with speech API
        app.run(host='0.0.0.0', port=5001, debug=True)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"Error: {e}")