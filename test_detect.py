from ultralytics import YOLO
import cv2

# load model - use our trained toy car model
model = YOLO("toycar_best.pt")

# load image (put a road image in same folder)
img = cv2.imread("road2.jpg")

# run detection with lower confidence for toy cars
results = model(img, conf=0.2)[0]

car_count = 0

# Draw all detections and count cars
for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
    x1, y1, x2, y2 = map(int, box)
    class_id = int(cls)
    class_name = model.names[class_id]
    confidence = float(conf)
    
    # Draw bounding box
    color = (0, 255, 0) if class_id == 0 else (255, 0, 0)  # Green for cars (class 0 in our model)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"{class_name} {confidence:.2f}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    print(f"Detected: {class_name} (conf: {confidence:.2f})")
    
    if class_id == 0:   # class 0 = car in our custom model
        car_count += 1

print(f"\nTotal cars detected: {car_count}")

# Save annotated image
cv2.imwrite("result.jpg", img)
print("Saved annotated image to result.jpg")

