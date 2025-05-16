from flask import Flask, request, jsonify
import os
import tempfile
import urllib.request
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Define class labels
CLASS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 
         "Normal", 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 64, 64

# Model URL - direct public access
MODEL_URL = "https://goutham469uploads.s3.amazonaws.com/uploads/1746122518697-densenet121_model5.h5"

# Define a dummy Cast layer to satisfy loading
class Cast(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# Global variable for model to avoid reloading
model = None

def load_model():
    """Load the TensorFlow model from public URL"""
    global model
    if model is None:
        try:
            # Create a temporary file to store the model
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                logger.info(f"Downloading model from {MODEL_URL}")
                urllib.request.urlretrieve(MODEL_URL, tmp.name)
                
                # Load the model with custom objects
                logger.info("Loading model...")
                model = tf.keras.models.load_model(
                    tmp.name,
                    custom_objects={'Cast': Cast}
                )
                logger.info("Model loaded successfully!")
            
            # Delete the temporary file
            os.unlink(tmp.name)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    return model

def download_video(video_url):
    """Download video from URL to a temporary file"""
    logger.info(f"Downloading video from URL: {video_url}")
    start_time = time.time()
    
    try:
        # Create a temporary file to store the video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            # Set timeout for download (10 seconds for connection, 300 seconds for read)
            urllib.request.urlretrieve(video_url, tmp.name)
            video_path = tmp.name
        
        download_time = time.time() - start_time
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
        logger.info(f"Video downloaded successfully in {download_time:.2f}s. Size: {file_size:.2f}MB")
        
        return video_path
    except urllib.error.URLError as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise Exception(f"Failed to download video: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error downloading video: {str(e)}")
        raise Exception(f"Unexpected error while downloading video: {str(e)}")

def process_video(video_path):
    """Process video file and make predictions"""
    logger.info(f"Processing video from {video_path}")
    
    # Verify the video file exists and is valid
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return {"error": "Video file not found"}
    
    # Check file size
    file_size = os.path.getsize(video_path)
    max_size = 100 * 1024 * 1024  # 100MB limit
    
    if file_size > max_size:
        logger.warning(f"Video size ({file_size} bytes) exceeds limit ({max_size} bytes)")
        return {"error": "Video file too large, max possible 100MB.", "statusCode": 413}
    
    # Read video using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video is valid and can be opened
    if not cap.isOpened():
        logger.error("Failed to open video file")
        return {"error": "Invalid video file or format"}
    
    frames = []
    frame_count = 0
    sample_rate = 5  # Process every 5th frame to reduce computation
    
    start_time = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Sample frames to reduce processing time
        if frame_count % sample_rate == 0:
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        frame_count += 1
        
        # Add a limit to handle very long videos
        if frame_count > 3000:  # Process at most 3000 frames
            logger.warning("Video exceeds 3000 frames, truncating")
            break
    
    cap.release()
    
    # Log extraction time
    extract_time = time.time() - start_time
    logger.info(f"Frame extraction completed in {extract_time:.2f}s. Extracted {len(frames)} frames from {frame_count} total frames")
    
    if not frames:
        logger.warning("No frames detected in the video")
        return {"error": "No frames detected in the video"}

    # Load model
    model = load_model()
    
    # Preprocess and predict
    logger.info(f"Processing {len(frames)} frames")
    frames_np = np.array(frames, dtype=np.float32)
    preprocessed_frames = tf.keras.applications.densenet.preprocess_input(frames_np)
    
    # Make predictions in batches
    batch_size = 32
    start_time = time.time()
    predictions = model.predict(preprocessed_frames, batch_size=batch_size)
    predict_time = time.time() - start_time
    logger.info(f"Prediction completed in {predict_time:.2f}s")
    
    # Count predictions
    print(predictions)

    prediction_counts = {}
    for pred in predictions:
        idx = np.argmax(pred)
        prediction_counts[idx] = prediction_counts.get(idx, 0) + 1

    print(prediction_counts)

    # Map index to class name and calculate percentages
    total_frames = len(frames)
    results = {}
    
    for idx, count in prediction_counts.items():
        class_name = CLASS[idx]
        percentage = (count / total_frames) * 100
        results[class_name] = {
            "count": int(count),
            "percentage": round(percentage, 2)
        }
    
    # Find dominant activity
    dominant_idx = max(prediction_counts.items(), key=lambda x: x[1])[0]
    dominant_activity = CLASS[dominant_idx]

    print( "video_path :- ", video_path )

    if video_path == "" :
        return {
        "dominant_activity": "abuse",
        "is_normal": dominant_activity == "abuse",
        "frame_count": frame_count,
        "processed_frames": total_frames,
        "classifications": results,
        "more":prediction_counts
    }
    
    return {
        "dominant_activity": dominant_activity,
        "is_normal": dominant_activity == "Normal",
        "frame_count": frame_count,
        "processed_frames": total_frames,
        "classifications": results,
        "more":prediction_counts
    }

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Handle CORS preflight requests
@app.route('/detect', methods=['OPTIONS'])
def options_handler():
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response, 200

# Main endpoint for violence detection
@app.route('/detect', methods=['POST'])
def detect_violence():
    """Process video detection request"""
    logger.info("Processing violence detection request")
    
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        # Get JSON data
        data = request.get_json()
        
        # Check if video URL is present
        if 'video_url' not in data:
            return jsonify({"error": "No video_url found in request"}), 400
        
        video_url = data['video_url']
        
        # Validate URL format
        if not video_url.startswith(('http://', 'https://')):
            return jsonify({"error": "Invalid video URL format"}), 400
            
        try:
            # Download video from URL
            video_path = download_video(video_url)
            
            # Process the video
            results = process_video(video_path)
            
            # Check if processing returned an error
            if 'error' in results:
                if 'statusCode' in results:
                    status_code = results.pop('statusCode')
                    return jsonify(results), status_code
                return jsonify(results), 400
            
            # Delete temporary file
            try:
                os.unlink(video_path)
            except:
                logger.warning(f"Failed to delete temporary file: {video_path}")
            
            # Return results
            response = jsonify(results)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return jsonify({"error": f"Error processing video: {str(e)}"}), 500
        
    except Exception as e:
        logger.error(f"Unhandled exception in detect_violence: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500



if __name__ == '__main__':
    # Preload the model when the app starts
    try:
        load_model()
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model at startup: {str(e)}")
        
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)