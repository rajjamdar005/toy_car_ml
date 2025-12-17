# 🚦 Smart Traffic Density Detection System

An IoT-based traffic management system using **ESP32-CAM** and **Custom Trained YOLOv8** for real-time toy car detection and traffic signal control.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)
![ESP32](https://img.shields.io/badge/ESP32--CAM-IoT-orange.svg)

## 📋 Overview

This project implements an intelligent traffic signal system that:
- Captures real-time images from ESP32-CAM
- Sends images to a cloud/local server
- Detects toy cars using a custom-trained YOLOv8 model
- Makes traffic signal decisions based on car count

### Traffic Signal Logic:
| Cars Detected | Signal |
|---------------|--------|
| 0-3 cars | 🔴 RED |
| 4+ cars | 🟢 GREEN |

## 🏗️ Architecture

```
┌─────────────┐     HTTP POST      ┌─────────────────┐
│  ESP32-CAM  │ ─────────────────► │  Flask Server   │
│  (Camera)   │                    │  (YOLO Model)   │
└─────────────┘ ◄───────────────── └─────────────────┘
                   JSON Response
                   {cars, decision}
```

## 📁 Project Structure

```
├── server_yolo.py          # Flask server with YOLO detection
├── toycar_best.pt          # Custom trained YOLOv8 model
├── traffic_client/
│   └── traffic_client.ino  # ESP32-CAM Arduino code
├── fix_and_retrain.py      # Model retraining script
├── train_model.py          # Original training script
├── prepare_training.py     # Data labeling tool
├── test_detect.py          # Model testing script
├── requirements.txt        # Python dependencies
└── dataset/
    └── toycar.yaml         # Dataset configuration
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/traffic-detection.git
cd traffic-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Server
```bash
python server_yolo.py
```

Server will start at `http://0.0.0.0:5000`

### 4. Upload ESP32 Code
1. Open `traffic_client/traffic_client.ino` in Arduino IDE
2. Update WiFi credentials and server IP
3. Upload to ESP32-CAM

## 🔧 Configuration

### Server Settings (server_yolo.py)
```python
CONFIDENCE = 0.3          # Detection confidence threshold
TRAFFIC_MIN_THRESHOLD = 4 # Minimum cars for GREEN signal
```

### ESP32 Settings (traffic_client.ino)
```cpp
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* serverUrl = "http://YOUR_SERVER_IP:5000/analyze";
```

## 📡 API Endpoints

### GET `/`
Health check endpoint
```json
{
  "status": "online",
  "model": "toycar_best.pt"
}
```

### POST `/analyze`
Analyze image for traffic density

**Request:** `multipart/form-data` with `image` file

**Response:**
```json
{
  "cars": 4,
  "decision": "GREEN",
  "confidence_scores": [0.85, 0.72, 0.68, 0.54]
}
```

## 🎯 Model Training

The model was trained on 216 labeled images of toy cars.

### Training Results:
| Metric | Value |
|--------|-------|
| mAP50 | 90.1% |
| mAP50-95 | 41.4% |
| Precision | 89.1% |
| Recall | 91.4% |

### Retrain the Model:
```bash
python fix_and_retrain.py
```

## 🌐 Deployment

### Deploy to Render.com:
1. Push code to GitHub
2. Create new Web Service on Render
3. Connect your repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `python server_yolo.py`

## 🛠️ Hardware Requirements

- **ESP32-CAM** (AI-Thinker module)
- **FTDI Programmer** (for uploading code)
- USB Cable
- Toy cars for testing 🚗

## 📊 Sample Output

```
🚦 TRAFFIC DENSITY DETECTION SERVER
    (Custom Trained YOLO Model)
==================================================
Model: toycar_best.pt
Threshold: 4+ cars = GREEN
Confidence: 0.3
==================================================

Cars: 4, Raw: GREEN, Final: GREEN
Cars: 2, Raw: RED, Final: RED
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Your Name**

---

⭐ Star this repo if you found it helpful!
