from ultralytics import YOLO
import cv2
import numpy as np

# Load models
coco_model = YOLO("models/yolov8n.pt")
license_plate_model = YOLO("runs/detect/train_detect_plate/weights/last.pt")
traffic_light_model = YOLO("runs/detect/train_traffic_light/weights/last.pt")

# Các lớp tương ứng với các loại xe
vehicle_classes = [2, 3, 5, 7]  # Xe máy, ô tô, xe tải, xe buýt

# Visualize results
def draw_results(img, results, names, color=(0, 255, 0), filter_classes=None):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        classes = result.boxes.cls.cpu().numpy()  # Class IDs
        confs = result.boxes.conf.cpu().numpy()  # Confidence scores
        for box, cls, conf in zip(boxes, classes, confs):
            if filter_classes and int(cls) not in filter_classes:
                continue  # Bỏ qua nếu lớp không nằm trong danh sách lọc
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
            label = f"{names[int(cls)]} {conf:.2f}"  # Thêm tên lớp và độ tin cậy
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Vẽ khung
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Ghi nhãn

# Mở video từ file
cap = cv2.VideoCapture('data/videos/sample2.mp4')

# Kiểm tra xem video có mở thành công không
if not cap.isOpened():
    print("Không thể mở video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình")
        break

    # Run inference with all models
    coco_result = coco_model.predict(frame)
    license_plate_result = license_plate_model.predict(frame)
    traffic_light_result = traffic_light_model.predict(frame)

    # Vẽ kết quả lên khung hình
    output_frame = frame.copy()
    
    # Vẽ kết quả của COCO model (xe máy, ô tô, xe tải, xe buýt)
    draw_results(output_frame, coco_result, coco_model.names, color=(255, 0, 0), filter_classes=vehicle_classes)
    
    # Vẽ kết quả của License Plate model
    draw_results(output_frame, license_plate_result, license_plate_model.names, color=(0, 255, 0))
    
    # Vẽ kết quả của Traffic Light model
    draw_results(output_frame, traffic_light_result, traffic_light_model.names, color=(0, 0, 255))

    # Hiển thị khung hình với kết quả phát hiện
    cv2.imshow("Real-time Detection", output_frame)

    # Nhấn 'q' để thoát khỏi video
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

# Giải phóng video và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
