"""
Traffic Density Detection Cloud Server
Detects small box-like objects (toy cars) using contour detection
Ignores white lane markers
"""

from flask import Flask, request, jsonify
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

# Decision history for smoothing (prevents flickering)
history = deque(maxlen=5)

# Traffic thresholds
TRAFFIC_MIN_THRESHOLD = 4   # 4+ cars = GREEN (enough traffic)
TRAFFIC_MAX_THRESHOLD = 5   # Upper limit for GREEN

# Image counter for naming
image_counter = 0

# Detection parameters (tuned for toy cars from overhead camera)
MIN_AREA = 300       # Minimum contour area (smaller for distant cars)
MAX_AREA = 20000     # Maximum contour area
MIN_ASPECT = 0.4     # Minimum aspect ratio (width/height)
MAX_ASPECT = 2.5     # Maximum aspect ratio (cars are roughly rectangular)


def detect_toy_cars(img):
    """
    Precisely detect silver/gray toy cars only
    Filters out: white lanes, dark background, noise
    Returns list of detected cars
    """
    # Convert to HSV and LAB for better color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Get image dimensions for relative sizing
    img_height, img_width = img.shape[:2]
    img_area = img_height * img_width
    
    # Dynamic area thresholds based on image size
    min_car_area = max(200, int(img_area * 0.001))   # At least 0.1% of image
    max_car_area = int(img_area * 0.15)               # At most 15% of image
    
    # STEP 1: Create mask for SILVER/GRAY toy cars
    # Silver/gray = low saturation (0-50), medium brightness (70-200)
    lower_silver = np.array([0, 0, 70])
    upper_silver = np.array([180, 50, 200])
    silver_mask = cv2.inRange(hsv, lower_silver, upper_silver)
    
    # STEP 2: Exclude WHITE areas (lane markers) - high brightness, low saturation
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # STEP 3: Exclude DARK areas (track surface) - low brightness
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Combine: silver areas minus white and dark
    car_mask = cv2.bitwise_and(silver_mask, cv2.bitwise_not(white_mask))
    car_mask = cv2.bitwise_and(car_mask, cv2.bitwise_not(dark_mask))
    
    # STEP 4: Morphological cleanup - remove noise, fill gaps
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)
    
    # Remove small noise
    car_mask = cv2.morphologyEx(car_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    # Fill gaps in car bodies
    car_mask = cv2.morphologyEx(car_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # STEP 5: Find contours
    contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # STEP 6: Filter to find only toy cars
    detected_cars = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Size filter
        if area < min_car_area or area > max_car_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Aspect ratio filter (cars are wider than tall or roughly square)
        aspect = float(w) / h if h > 0 else 0
        if aspect < 0.5 or aspect > 2.5:
            continue
        
        # Solidity filter (how much contour fills its bounding box)
        rect_area = w * h
        solidity = area / rect_area if rect_area > 0 else 0
        if solidity < 0.45:  # Cars should be fairly solid shapes
            continue
        
        # Minimum dimension filter (too thin = not a car)
        if w < 15 or h < 15:
            continue
        
        # Convexity check - cars are fairly convex
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        if convexity < 0.6:
            continue
        
        # This is a toy car!
        detected_cars.append({
            'box': (x, y, w, h),
            'area': area,
            'solidity': solidity,
            'convexity': convexity
        })
    
    return detected_cars


def save_result_image(img, detected_cars, car_count, decision):
    """Save annotated image to logs directory"""
    global image_counter
    image_counter += 1
    
    # Create annotated image
    annotated = img.copy()
    
    # Draw all detected cars
    for i, det in enumerate(detected_cars):
        x, y, w, h = det['box']
        # Draw cyan rectangle for cars
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 255, 0), 2)
        # Label
        label = f"Car {i+1}"
        cv2.putText(annotated, label, (x, y-5), 
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
        message="Traffic Detection Server Ready (Contour Detection)",
        logs_dir=LOGS_DIR,
        images_processed=image_counter,
        green_range=f"{TRAFFIC_MIN_THRESHOLD}-{TRAFFIC_MAX_THRESHOLD} cars"
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze image for toy cars (precise silver/gray detection)
    
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
        
        # Detect toy cars (precise detection)
        detected_cars = detect_toy_cars(img)
        car_count = len(detected_cars)
        
        # Determine decision based on car count
        # 4-5 cars = GREEN (normal traffic), else = RED
        if car_count >= TRAFFIC_MIN_THRESHOLD and car_count <= TRAFFIC_MAX_THRESHOLD:
            decision = "GREEN"
        else:
            decision = "RED"
        
        # Add to history for smoothing
        history.append(decision)
        
        # Final decision = majority vote from last 5
        final_decision = Counter(history).most_common(1)[0][0]
        
        # Save result image for debugging
        saved_filename = save_result_image(img, detected_cars, car_count, final_decision)
        
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
    global TRAFFIC_MIN_THRESHOLD, TRAFFIC_MAX_THRESHOLD, MIN_AREA, MAX_AREA
    
    if request.method == "POST":
        data = request.get_json() or {}
        if "min_threshold" in data:
            TRAFFIC_MIN_THRESHOLD = int(data["min_threshold"])
        if "max_threshold" in data:
            TRAFFIC_MAX_THRESHOLD = int(data["max_threshold"])
        if "min_area" in data:
            MIN_AREA = int(data["min_area"])
        if "max_area" in data:
            MAX_AREA = int(data["max_area"])
    
    return jsonify(
        min_threshold=TRAFFIC_MIN_THRESHOLD,
        max_threshold=TRAFFIC_MAX_THRESHOLD,
        min_area=MIN_AREA,
        max_area=MAX_AREA,
        history_size=len(history)
    )


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚦 TRAFFIC DENSITY DETECTION SERVER")
    print("    (Contour Detection Mode)")
    print("="*50)
    print(f"Threshold: {TRAFFIC_MIN_THRESHOLD}-{TRAFFIC_MAX_THRESHOLD} cars = GREEN, else RED")
    print(f"Min Area: {MIN_AREA}, Max Area: {MAX_AREA}")
    print(f"Logs: {LOGS_DIR}")
    print("="*50 + "\n")
    
    # Run server (use 0.0.0.0 to accept connections from ESP32)
    app.run(host="0.0.0.0", port=5000, debug=False)
