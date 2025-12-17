"""
Test script to send image to the traffic detection server
"""
import requests

# Server URL
SERVER_URL = "http://127.0.0.1:5000/analyze"

# Send image
with open("road2.jpg", "rb") as f:
    response = requests.post(SERVER_URL, files={"image": f})

if response.status_code == 200:
    data = response.json()
    print(f"🚗 Cars detected: {data['cars']}")
    print(f"🚦 Decision: {data['decision']}")
    print(f"📊 Raw decision: {data.get('raw_decision', 'N/A')}")
    if data.get('detections'):
        print("\nDetections:")
        for det in data['detections']:
            print(f"  - {det['class']}: {det['confidence']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
