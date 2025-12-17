"""
Traffic Density Detection Cloud Server
Uses CUSTOM TRAINED YOLO model for precise toy car detection
"""

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque, Counter
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create logs directory for saving results (absolute path)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
logger.info(f"Results will be saved to: {LOGS_DIR}")

# Create log file
log_file = os.path.join(LOGS_DIR, "detection_log.txt")

app = Flask(__name__)

# Load CUSTOM TRAINED YOLO model for toy cars
MODEL_PATH = os.path.join(SCRIPT_DIR, "toycar_best.pt")
logger.info(f"Loading custom toy car model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
logger.info("Model loaded successfully!")

# Decision history for smoothing (prevents flickering)
# Reduced to 3 for faster response
history = deque(maxlen=3)

# Traffic thresholds
TRAFFIC_MIN_THRESHOLD = 4   # 4+ cars = GREEN (enough traffic)
TRAFFIC_MAX_THRESHOLD = 5   # Upper limit for GREEN

# Image counter for naming
image_counter = 0

# Detection confidence threshold
CONFIDENCE = 0.3


def save_result_image(img, results, car_count, decision):
    """Save annotated image to logs directory"""
    global image_counter
    image_counter += 1
    
    # Create annotated image
    annotated = img.copy()
    
    # Draw all detected cars
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # Draw cyan rectangle for cars
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
        label = f"Car {i+1} {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Add decision overlay
    overlay_color = (0, 0, 255) if decision == "RED" else (0, 255, 0)
    cv2.rectangle(annotated, (5, 5), (200, 70), (0, 0, 0), -1)
    cv2.putText(annotated, f"Cars: {car_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Decision: {decision}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{image_counter:04d}_{timestamp}_{decision}_cars{car_count}.jpg"
    filepath = os.path.join(LOGS_DIR, filename)
    
    # Save image
    cv2.imwrite(filepath, annotated)
    logger.info(f"Saved: {filename}")
    
    # Append to log file
    with open(log_file, "a") as f:
        f.write(f"{timestamp} | Image #{image_counter} | Cars: {car_count} | Decision: {decision}\n")
    
    return filename


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify(
        status="online", 
        message="Traffic Detection Server (Custom YOLO Model)",
        model=MODEL_PATH,
        logs_dir=LOGS_DIR,
        images_processed=image_counter,
        green_range=f"{TRAFFIC_MIN_THRESHOLD}-{TRAFFIC_MAX_THRESHOLD} cars"
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze image for toy cars using CUSTOM TRAINED YOLO model
    
    Expects: multipart/form-data with 'image' file
    Returns: JSON with car count and traffic decision
    """
    try:
        # Check if image was sent
        if "image" not in request.files:
            return jsonify(error="No image provided"), 400
        
        file = request.files["image"]
        
        # Decode image from bytes
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify(error="Invalid image"), 400
        
        logger.info(f"Received image: {img.shape}")
        
        # Run YOLO detection with custom model
        results = model(img, conf=CONFIDENCE, verbose=False)[0]
        
        # Count detected cars (class 0 = car in our custom model)
        car_count = len(results.boxes)
        
        # Determine decision based on car count
        # 4+ cars = GREEN (enough traffic to justify green signal)
        # 1-3 cars = RED (not enough traffic)
        # 0 cars = RED (no traffic)
        if car_count >= 4:
            decision = "GREEN"
        else:
            decision = "RED"
        
        # Add to history for smoothing
        history.append(decision)
        
        # Final decision = majority vote from last 5
        final_decision = Counter(history).most_common(1)[0][0]
        
        # Save result image for debugging
        saved_filename = save_result_image(img, results, car_count, final_decision)
        
        logger.info(f"Cars: {car_count}, Raw: {decision}, Final: {final_decision}")
        
        return jsonify(
            cars=car_count,
            decision=final_decision,
            raw_decision=decision
        )
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify(error=str(e)), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Reset decision history"""
    history.clear()
    return jsonify(status="History cleared")


@app.route("/config", methods=["GET", "POST"])
def config():
    """Get or update configuration"""
    global TRAFFIC_MIN_THRESHOLD, TRAFFIC_MAX_THRESHOLD, CONFIDENCE
    
    if request.method == "POST":
        data = request.get_json() or {}
        if "min_threshold" in data:
            TRAFFIC_MIN_THRESHOLD = int(data["min_threshold"])
        if "max_threshold" in data:
            TRAFFIC_MAX_THRESHOLD = int(data["max_threshold"])
        if "confidence" in data:
            CONFIDENCE = float(data["confidence"])
    
    return jsonify(
        min_threshold=TRAFFIC_MIN_THRESHOLD,
        max_threshold=TRAFFIC_MAX_THRESHOLD,
        confidence=CONFIDENCE,
        history_size=len(history)
    )


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚦 TRAFFIC DENSITY DETECTION     python server_yolo.py    python server_yolo.pySERVER")
    print("    (Custom Trained YOLO Model)")
    print("="*50)
    print(f"Model: {MODEL_PATH}")
    print(f"Threshold: {TRAFFIC_MIN_THRESHOLD}-{TRAFFIC_MAX_THRESHOLD} cars = GREEN")
    print(f"Confidence: {CONFIDENCE}")
    print(f"Logs: {LOGS_DIR}")
    print("="*50 + "\n")
    
    # Run server (use 0.0.0.0 to accept connections from ESP32)
    app.run(host="0.0.0.0", port=5000, debug=False)

