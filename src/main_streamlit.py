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

# Thêm vào đầu file, sau phần khai báo biến
previous_positions = {}  # Lưu lịch sử vị trí của các vehicle
vehicle_directions = {}  # Lưu hướng di chuyển của các vehicle

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

# Add these functions before draw_results

def track_vehicle_direction(vehicle_id, center_x, center_y, min_movement=10):
    """Track vehicle movement and determine direction"""
    if vehicle_id not in previous_positions:
        previous_positions[vehicle_id] = []
        
    positions = previous_positions[vehicle_id]
    current_pos = (center_x, center_y)
    
    if not positions or math.dist(current_pos, positions[-1]) > min_movement:
        positions.append(current_pos)
    
    if len(positions) > 5:  # Keep last 5 positions
        positions.pop(0)
        
    if len(positions) >= 2:
        first_pos = positions[0]
        last_pos = positions[-1]
        
        dx = last_pos[0] - first_pos[0]  # Fixed: access tuple elements properly
        dy = last_pos[1] - first_pos[1]
        
        # Calculate movement angle
        angle = math.degrees(math.atan2(dy, dx))
        
        # Determine primary direction based on angle
        if -45 <= angle <= 45 or angle >= 135 or angle <= -135:
            direction = "horizontal"
        else:
            direction = "vertical"
            
        vehicle_directions[vehicle_id] = direction
        return direction
    return None

def calculate_movement_vector(vehicle_id, center_x, center_y):
    """Calculate movement vector from position history"""
    if vehicle_id in previous_positions and len(previous_positions[vehicle_id]) >= 2:
        positions = previous_positions[vehicle_id]
        start = positions[0]
        end = positions[-1]
        return (end[0] - start[0], end[1] - start[1])
    return (0, 0)

def calculate_stop_line_angle(stop_line_points):
    """Calculate angle of stop line"""
    # Get first and last points of stop line
    p1 = stop_line_points[0]
    p2 = stop_line_points[-1]
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    return math.degrees(math.atan2(dy, dx))

def calculate_intersection_angle(movement_vector, stop_line_angle):
    """Calculate angle between movement vector and stop line"""
    movement_angle = math.degrees(math.atan2(movement_vector[1], movement_vector[0]))
    intersection_angle = abs(movement_angle - stop_line_angle)
    
    # Normalize angle to 0-180 range
    if intersection_angle > 180:
        intersection_angle = 360 - intersection_angle
    return intersection_angle

# Update draw_results function to use movement vectors
def draw_results(img, vehicle_tracks, assigned_plates, traffic_lights, vehicle_detections, traffic_lights_data):
    violation_img = img.copy()
    
    # Get stop line angle first
    stop_line_angle = None
    for light_id, data in traffic_lights_data.items():
        stop_line = data["stop_line"]
        stop_line_angle = calculate_stop_line_angle(stop_line)
        break  # Assuming single stop line
    
    # Process vehicles and track movement
    for track in vehicle_tracks:
        x1, y1, x2, y2, vehicle_id = map(int, track[:5])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Get movement direction
        direction = track_vehicle_direction(vehicle_id, center_x, center_y)
        dx, dy = calculate_movement_vector(vehicle_id, center_x, center_y)
        
        # Draw movement vector
        if dx != 0 or dy != 0:
            vector_scale = 2.0
            end_x = int(center_x + dx * vector_scale)
            end_y = int(center_y + dy * vector_scale)
            cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), 
                          (255, 0, 0), 2, tipLength=0.3)
            
            # Calculate and display intersection angle
            if stop_line_angle is not None:
                intersection_angle = calculate_intersection_angle((dx, dy), stop_line_angle)
                angle_text = f"Angle: {intersection_angle:.1f}°"
                cv2.putText(img, angle_text, (x1, y1-35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Only check violations for vertical movement (downward)
        is_violating = False
        if direction == "vertical" and dy > 0:  # Only check downward movement
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
                            if light_cls == 1:  # Red light
                                is_violating = True
                                break

        # Rest of the drawing_results function remains the same...
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
                        if light_cls == 1:  # Red light
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
                                "vehicle_class": class_name,
                                "vehicle_id": vehicle_id,
                                "direction": vehicle_directions.get(vehicle_id, "unknown"),
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

def calculate_direction(previous_pos, current_pos, min_movement=10):
    """
    Calculate movement direction based on y-axis movement
    Returns: "down", "up", or "unknown"
    """
    if not previous_pos:
        return "unknown"
        
    dy = current_pos[1] - previous_pos[1]
    
    # Only consider significant movement
    if abs(dy) < min_movement:
        return "unknown"
        
    return "down" if dy > 0 else "up"

def calculate_movement_angle(positions):
    """
    Calculate movement angle from position history
    Returns angle in degrees (-180 to 180)
    """
    if len(positions) < 2:
        return None
        
    start = positions[0]
    end = positions[-1]
    
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    # Calculate angle in degrees
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def is_target_direction(angle, target_angle=90, threshold=30):
    """
    Check if movement angle is within threshold of target angle
    Default target_angle=90 (downward movement)
    """
    if angle is None:
        return False
        
    diff = abs((angle - target_angle + 180) % 360 - 180)
    return diff <= threshold

def track_vehicle_movement(vehicle_id, center_point, min_movement=10):
    """
    Track vehicle movement and calculate direction angle
    """
    if vehicle_id not in previous_positions:
        previous_positions[vehicle_id] = []
    
    positions = previous_positions[vehicle_id]
    
    if not positions or math.dist(center_point, positions[-1]) > min_movement:
        positions.append(center_point)
        
    if len(positions) > 10:  # Keep last 10 positions
        positions.pop(0)
        
    angle = calculate_movement_angle(positions)
    return angle

def track_vehicle_direction(vehicle_id, center_x, center_y):
    """Track vehicle movement and determine direction"""
    if vehicle_id not in previous_positions:
        previous_positions[vehicle_id] = []
        
    positions = previous_positions[vehicle_id]
    positions.append((center_x, center_y))
    
    if len(positions) > 5:  # Keep last 5 positions
        positions.pop(0)
        
    if len(positions) >= 2:
        first_pos = positions[0]
        last_pos = positions[-1]
        
        dx = last_pos[0] - first_pos
        dy = last_pos[1] - first_pos
        
        # Get primary direction based on larger movement
        if abs(dx) > abs(dy):
            direction = "horizontal"
        else:
            direction = "vertical"
            
        vehicle_directions[vehicle_id] = direction
        return direction
    return None
