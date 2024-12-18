import json
import cv2
from ultralytics import YOLO
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime
from paddleocr import PaddleOCR
from sort import Sort
import torch


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# PaddleOCR initialization
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Check GPU availability
def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available. Using:", torch.cuda.get_device_name(0))
        return "cuda"
    else:
        print("GPU is not available. Using CPU.")
        return "cpu"

# Select device
device = check_gpu()

# Load YOLO models
coco_model = YOLO("models/yolov8n.pt").to(device)  # Vehicle detection
license_plate_model = YOLO("runs/detect/train_detect_plate/weights/last.pt").to(device)  # License plate detection
traffic_light_status_model = YOLO("runs/detect/train_traffic_light_status/weights/last.pt").to(device)  # Traffic light detection

# Initialize tracker
tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.4)

# Define classes
vehicle_classes = [2, 3, 5, 7]

vehicle_class_names = {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}
traffic_light_class_names = {0: "Green", 1: "Red"}

# Function for OCR
def paddle_ocr(frame, x1, y1, x2, y2):
    plate_frame = frame[y1:y2, x1:x2]
    height, width = plate_frame.shape[:2]
    aspect_ratio = height / width
    text = ""

    if aspect_ratio > 0.5:
        upper_frame = plate_frame[:height // 2, :]
        lower_frame = plate_frame[height // 2:, :]
        upper_result = ocr.ocr(upper_frame, det=False, rec=True, cls=False)
        lower_result = ocr.ocr(lower_frame, det=False, rec=True, cls=False)

        upper_text = "".join(r[0][0] for r in upper_result if int(r[0][1] * 100) > 80)
        lower_text = "".join(r[0][0] for r in lower_result if int(r[0][1] * 100) > 80)
        text = upper_text + lower_text
    else:
        result = ocr.ocr(plate_frame, det=False, rec=True, cls=False)
        text = "".join(r[0][0] for r in result if int(r[0][1] * 100) > 80)

    text = re.sub(r'[^\w]', '', text).replace("O", "0")
    vietnam_plate_pattern = re.compile(r'^\d{2}[A-Z]-?\d{3,5}$|^[A-Z]{1,2}-?\d{4,5}$')
    return text if vietnam_plate_pattern.match(text) else "Invalid"

# Lưu trữ biển số có độ tin cậy cao nhất cho mỗi phương tiện
vehicle_plate_history = {}

def assign_plate_to_vehicle(vehicle_tracks, plates, frame):
    assigned_plates = {}
    for plate_box in plates:
        px1, py1, px2, py2 = map(int, plate_box[:4])  # Ensure each plate_box is an individual box
        plate_text = paddle_ocr(frame, px1, py1, px2, py2)

        # Kiểm tra độ tin cậy của biển số
        plate_confidence = 0  # Giả sử độ tin cậy ban đầu là 0
        if plate_text != "Invalid":
            # Đoạn mã này có thể được thay đổi nếu bạn có cách đo độ tin cậy của OCR
            plate_confidence = max([r[0][1] for r in ocr.ocr(frame[py1:py2, px1:px2], det=False, rec=True, cls=False)]) 

        plate_center = ((px1 + px2) // 2, (py1 + py2) // 2)
        for vehicle in vehicle_tracks:
            vx1, vy1, vx2, vy2, vehicle_id = map(int, vehicle[:5])
            if vx1 <= plate_center[0] <= vx2 and vy1 <= plate_center[1] <= vy2:
                # Nếu chưa có biển số cho phương tiện này hoặc biển số mới có độ tin cậy cao hơn
                if vehicle_id not in vehicle_plate_history:
                    vehicle_plate_history[vehicle_id] = (plate_text, plate_confidence)
                else:
                    prev_plate, prev_confidence = vehicle_plate_history[vehicle_id]
                    if plate_confidence > prev_confidence:  # Cập nhật nếu độ tin cậy cao hơn
                        vehicle_plate_history[vehicle_id] = (plate_text, plate_confidence)

                assigned_plates[vehicle_id] = (plate_box, vehicle_plate_history[vehicle_id][0])
                break
    return assigned_plates


# Load traffic lights data
with open("json/traffic_lights1.json", "r") as file:
    traffic_lights_data = json.load(file)
violating_vehicles = {}  # Dictionary to store the ID and status (Red Box) of violating vehicles

# Add these at the top of the file with other imports
violation_save_path = "results/violations"
violation_json_path = os.path.join(violation_save_path, "violating_vehicles.json")

# Create violations directory if it doesn't exist
os.makedirs(violation_save_path, exist_ok=True)

# Add at the top of the file, after other imports
violation_images_path = os.path.join(violation_save_path, "images")
os.makedirs(violation_images_path, exist_ok=True)  # Create images subdirectory

def save_violation_data(violation_data):
    """Save violation data to JSON file"""
    existing_data = {}
    if os.path.exists(violation_json_path):
        with open(violation_json_path, 'r') as f:
            existing_data = json.load(f)
    
    # Update existing data with new violation data
    existing_data.update(violation_data)
    
    with open(violation_json_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

        #save dâta to SQL database
    save_to_databases(violation_data)

def initialize_database():
    conn = sqlite3.connect('ViolatingDatabase.db')
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS LicensePlates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            license_plate TEXT,
            vehicle_class TEXT,
            time_stamp TEXT,
            status TEXT,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_databases(violation_data):
    try:
        # Initialize database if not exists
        initialize_database()
        
        conn = sqlite3.connect('ViolatingDatabase.db')
        cursor = conn.cursor()
        
        # Get the first (and only) vehicle ID from the dictionary
        vehicle_id = list(violation_data.keys())[0]
        data = violation_data[vehicle_id]
        
        cursor.execute(
            '''
            INSERT INTO LicensePlates(
                vehicle_id, 
                license_plate, 
                vehicle_class, 
                time_stamp, 
                status, 
                image_path
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ''', 
            (
                data["vehicle_id"],
                data["plate_number"],
                data["vehicle_class"],
                data["timestamp"],
                data["status"],
                data["image_path"]
            )
        )
        conn.commit()
        conn.close()
        print(f"Successfully saved violation data for vehicle {vehicle_id}")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error saving violation data: {e}")

# First, modify vehicle tracking to include class information
def draw_results(img, vehicle_tracks, assigned_plates, traffic_lights, vehicle_detections, traffic_lights_data):
    violation_img = img.copy()
    
    # Create mapping of vehicle IDs to their classes
    vehicle_classes_map = {}
    for detection in vehicle_detections:
        x1, y1, x2, y2, cls, conf = detection
        # Store class info when creating detections
        vehicle_classes_map[tuple([x1, y1, x2, y2])] = cls

    for track in vehicle_tracks:
        x1, y1, x2, y2, vehicle_id = map(int, track[:5])
        is_violating = False
        
        # Get vehicle class by matching track coordinates with detections
        vehicle_class = None
        for det_coords, cls in vehicle_classes_map.items():
            if abs(det_coords[0] - x1) < 10 and abs(det_coords[1] - y1) < 10:
                vehicle_class = int(cls)
                break
        
        class_name = vehicle_class_names.get(vehicle_class, "Unknown")

        for light_id, data in traffic_lights_data.items():
            stop_line = data["stop_line"]
            stop_line_y = min(p[1] for p in stop_line)
            
            if y2 >= stop_line_y and y1 <= stop_line_y:
                roi = data["roi"]
                roi_x, roi_y, roi_w, roi_h = roi
                
                for light in traffic_lights:
                    light_x1, light_y1, light_x2, light_y2, light_cls = map(int, light[:5])
                    if (light_x1 >= roi_x and light_x2 <= roi_x + roi_w and 
                        light_y1 >= roi_y and light_y2 <= roi_y + roi_h):
                        if light_cls == 0:  # Red light
                            is_violating = True
                            break
                
                if is_violating:
                    # Get best historical plate for this vehicle
                    best_plate_text = "Unknown"
                    best_plate_confidence = 0.0
                    
                    if vehicle_id in vehicle_plate_history:
                        best_plate_text, best_plate_confidence = vehicle_plate_history[vehicle_id]
                    elif vehicle_id in assigned_plates:
                        _, current_plate_text = assigned_plates[vehicle_id][:2]
                        best_plate_text = current_plate_text

                    if vehicle_id not in violating_vehicles:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Draw violation box and info on violation_img
                        cv2.rectangle(violation_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
                        violation_text = f"ID: {vehicle_id} | Class: {class_name} | Plate: {best_plate_text}"
                        cv2.putText(violation_img, violation_text, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Draw stop line and traffic light ROI
                        for light_id, data in traffic_lights_data.items():
                            stop_line = data["stop_line"]
                            for i in range(len(stop_line) - 1):
                                cv2.line(violation_img, tuple(stop_line[i]), tuple(stop_line[i + 1]), (0, 255, 0), 2)
                        
                        img_name = f"violation_id{vehicle_id}_{timestamp}.jpg"
                        img_path = os.path.join(violation_images_path, img_name)
                        cv2.imwrite(img_path, violation_img)
                        
                        violation_data = {
                            str(vehicle_id): {
                                "timestamp": timestamp,
                                "plate_number": best_plate_text,
                                "plate_confidence": float(best_plate_confidence),
                                "image_path": img_path,
                                "vehicle_class": class_name,  # Now using correct class name
                                "vehicle_id": vehicle_id,
                                "status": "violating_vehicle"
                            }
                        }
                        
                        save_violation_data(violation_data)
                        violating_vehicles[vehicle_id] = {
                            "status": "violating_vehicle",
                            "plate": best_plate_text,
                            "confidence": best_plate_confidence,
                            "timestamp": timestamp,
                            "class": class_name  # Added class info to violating vehicles
                        }
                break

        # Draw vehicle ID and box
        color_violating = (0, 0, 255) if is_violating or vehicle_id in violating_vehicles else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_violating, 2)
        
        # Display vehicle ID and best plate if available
        display_text = f"ID: {vehicle_id}"
        if vehicle_id in vehicle_plate_history:
            plate_text = vehicle_plate_history[vehicle_id][0]
        cv2.rectangle(img, (x1, y1 - 20), (x1 + 70, y1), color_violating, -1)
        cv2.putText(img, display_text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Rest of the drawing code remains the same 
    # Drawing plates - Modified this part
    for vehicle_id, plate_info in assigned_plates.items():
        if len(plate_info) >= 2:  # Make sure we have at least box and text
            plate_box, plate_text = plate_info[:2]
            px1, py1, px2, py2 = map(int, plate_box[:4])
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 255), 1)
            cv2.putText(img, plate_text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    for light in traffic_lights:
        x1, y1, x2, y2, cls = map(int, light[:5])
        color_light = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Đèn xanh thì là xanh, đèn đỏ là đỏ
        cv2.rectangle(img, (x1, y1), (x2, y2), color_light, 1)
        traffic_light_name = traffic_light_class_names.get(cls)
        cv2.putText(img, traffic_light_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_light, 2)

    for light_id, data in traffic_lights_data.items():
        roi = data["roi"]
        stop_line = data["stop_line"]
        x, y, w, h = roi
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for i in range(len(stop_line) - 1):
            cv2.line(img, tuple(stop_line[i]), tuple(stop_line[i + 1]), (0, 255, 0), 2)

# Main video processing
cap = cv2.VideoCapture('data/videos/sample2.mp4')
if not cap.isOpened():
    print("Cannot open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    coco_results = coco_model.predict(frame, stream=True, device=device)
    license_plate_results = license_plate_model.predict(frame, stream=True, device=device)
    traffic_light_results = traffic_light_status_model.predict(frame, stream=True, device=device)

    vehicle_detections = [
        [*map(int, box), int(cls), float(conf)]
        for result in coco_results for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy())
        if int(cls) in vehicle_classes
    ]

    vehicle_tracks = tracker.update(np.array(vehicle_detections)) if vehicle_detections else []
    plate_boxes = [box for result in license_plate_results for box in result.boxes.xyxy.cpu().numpy()]

    traffic_lights = [[*box, int(cls)] for result in traffic_light_results for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy())]
    assigned_plates = assign_plate_to_vehicle(vehicle_tracks, plate_boxes, frame)

    draw_results(frame, vehicle_tracks, assigned_plates, traffic_lights, vehicle_detections, traffic_lights_data)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()
