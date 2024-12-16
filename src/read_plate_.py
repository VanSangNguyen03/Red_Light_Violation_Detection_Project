#Import All the Required Libraries
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#Create a Video Capture Object
cap = cv2.VideoCapture("data/videos/sample2.mp4")
#Initialize the YOLOv10 Model
model = YOLO("runs/detect/train_detect_plate/weights/last.pt")
#Initialize the frame count
count = 0
#Class Names
className = ["plate_number"]
#Initialize the Paddle OCR
ocr = PaddleOCR(use_angle_cls = True, use_gpu = False)

def paddle_ocr(frame, x1, y1, x2, y2):
    # Cắt vùng biển số
    plate_frame = frame[y1:y2, x1:x2]

    # Xác định tỷ lệ chiều cao so với chiều rộng
    height, width = plate_frame.shape[:2]
    aspect_ratio = height / width

    # Nếu tỷ lệ chiều cao lớn (biển số hai dòng)
    if aspect_ratio > 0.5:  # Ngưỡng có thể điều chỉnh
        # Tách biển số thành hai vùng
        upper_frame = plate_frame[:height // 2, :]  # Dòng trên
        lower_frame = plate_frame[height // 2:, :]  # Dòng dưới

        # OCR từng phần
        upper_result = ocr.ocr(upper_frame, det=False, rec=True, cls=False)
        lower_result = ocr.ocr(lower_frame, det=False, rec=True, cls=False)

        # Xử lý dòng trên
        upper_text = ""
        for r in upper_result:
            if int(r[0][1] * 100) > 80:  # Ngưỡng tin cậy
                upper_text = r[0][0]

        # Xử lý dòng dưới
        lower_text = ""
        for r in lower_result:
            if int(r[0][1] * 100) > 80:  # Ngưỡng tin cậy
                lower_text = r[0][0]

        # Gộp lại hai dòng
        text = upper_text + lower_text
    else:
        # Biển số một dòng
        result = ocr.ocr(plate_frame, det=False, rec=True, cls=False)
        text = ""
        for r in result:
            if int(r[0][1] * 100) > 80:  # Ngưỡng tin cậy
                text = r[0][0]

    # Làm sạch ký tự OCR
    text = re.sub(r'[^\w]', '', text).replace("O", "0")

    # Kiểm tra và chuẩn hóa định dạng biển số
    vietnam_plate_pattern = re.compile(
        r'^\d{2}[A-Z]-?\d{3,5}$|^[A-Z]{1,2}-?\d{4,5}$'
    )
    if not vietnam_plate_pattern.match(text):
        return "Invalid"

    return text


def save_json(license_plates, startTime, endTime):
    # Generate individual JSON files for each 20-second interval
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plate": list(license_plates)
    }
    interval_file_path = "json/output_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent=2)

    # Cumulative JSON File
    cummulative_file_path = "json/LicensePlateData.json"

    # Kiểm tra nếu file tồn tại và có dữ liệu hợp lệ
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"File {cummulative_file_path} không hợp lệ, tạo file mới.")
                existing_data = []  # Khởi tạo dữ liệu trống nếu file không hợp lệ
    else:
        existing_data = [] 

    #Add new intervaal data to cummulative data
    existing_data.append(interval_data)

    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent = 2)

    #Save data to SQL database
    save_to_database(license_plates, startTime, endTime)
def save_to_database(license_plates, start_time, end_time):
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    for plate in license_plates:
        cursor.execute('''
            INSERT INTO LicensePlates(start_time, end_time, license_plate)
            VALUES (?, ?, ?)
        ''', (start_time.isoformat(), end_time.isoformat(), plate))
    conn.commit()
    conn.close()
startTime = datetime.now()
license_plates = set()
while True:
    ret, frame = cap.read()
    if ret:
        currentTime = datetime.now()
        count += 1
        print(f"Frame Number: {count}")
        results = model.predict(frame, conf = 0.45)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                classNameInt = int(box.cls[0])
                clsName = className[classNameInt]
                conf = math.ceil(box.conf[0]*100)/100
                #label = f'{clsName}:{conf}'
                label = paddle_ocr(frame, x1, y1, x2, y2)
                if label:
                    license_plates.add(label)
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
        if (currentTime - startTime).seconds >= 20:
            endTime = currentTime
            save_json(license_plates, startTime, endTime)
            startTime = currentTime
            license_plates.clear()
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
