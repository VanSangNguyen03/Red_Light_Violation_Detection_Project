import cv2
import json
import os
import numpy as np

traffic_lights_data = {}  # Cấu trúc dữ liệu lưu thông tin

# Biến tạm để lưu trạng thái
current_light_id = None
roi = [None, None]
stop_line_points = []
frame_copy = None

def select_roi(event, x, y, flags, param):
    global roi, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:  # Nhấp chuột trái để chọn điểm đầu
        roi[0] = (x, y)
        print(f"Điểm đầu ROI: {roi[0]}")
    elif event == cv2.EVENT_LBUTTONUP:  # Nhả chuột để chọn điểm cuối
        roi[1] = (x, y)
        print(f"Điểm cuối ROI: {roi[1]}")
        if roi[0] and roi[1]:
            cv2.rectangle(frame_copy, roi[0], roi[1], (0, 255, 0), 2)
            cv2.imshow("Chọn ROI đèn giao thông", frame_copy)

def draw_stop_line(event, x, y, flags, param):
    global stop_line_points, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:  # Nhấp chuột để thêm điểm
        stop_line_points.append((x, y))
        print(f"Thêm điểm vạch dừng: {x}, {y}")
        cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
        if len(stop_line_points) > 1:
            cv2.line(frame_copy, stop_line_points[-2], stop_line_points[-1], (0, 255, 0), 2)
        cv2.imshow("Vẽ vạch dừng", frame_copy)
    elif event == cv2.EVENT_RBUTTONDOWN:  # Nhấp chuột phải để hoàn tất
        if len(stop_line_points) > 1:
            pts = np.array(stop_line_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_copy, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
        cv2.imshow("Vẽ vạch dừng", frame_copy)
        print("Hoàn tất vẽ vạch dừng. Nhấn phím bất kỳ để lưu dữ liệu.")

def get_roi(frame, light_id):
    global roi, frame_copy
    roi = [None, None]
    frame_copy = frame.copy()

    cv2.namedWindow("Chọn ROI đèn giao thông", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Chọn ROI đèn giao thông", frame.shape[1], frame.shape[0])
    cv2.imshow("Chọn ROI đèn giao thông", frame_copy)
    cv2.setMouseCallback("Chọn ROI đèn giao thông", select_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if roi[0] is None or roi[1] is None:
        print("Vui lòng chọn cả hai điểm để xác định ROI.")
        return None

    x, y, w, h = roi[0][0], roi[0][1], roi[1][0] - roi[0][0], roi[1][1] - roi[0][1]
    return [x, y, w, h]

def get_stop_line(frame, light_id):
    global stop_line_points, frame_copy
    stop_line_points = []
    frame_copy = frame.copy()

    cv2.namedWindow("Vẽ vạch dừng", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vẽ vạch dừng", frame.shape[1], frame.shape[0])
    cv2.imshow("Vẽ vạch dừng", frame_copy)
    cv2.setMouseCallback("Vẽ vạch dừng", draw_stop_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(stop_line_points) < 2:
        print("Vui lòng chọn ít nhất hai điểm để vẽ vạch dừng.")
        return []

    return stop_line_points

def save_to_json(data, file_name="json/traffic_lights1.json"):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as file:
        json.dump(data, file, separators=(',', ':'), indent=4)
    print(f"Dữ liệu đã được lưu vào {file_name}.")

video_path = "data/videos/sample2.mp4"  # Thay bằng đường dẫn video của bạn
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    light_id = input("Nhập ID đèn giao thông (vd: light_1, hoặc 't' để thoát): ")
    if light_id.lower() == "t":
        break

    roi = get_roi(frame, light_id)
    if roi is None:
        print("Không chọn được ROI. Thử lại.")
        continue

    stop_line = get_stop_line(frame, light_id)
    if not stop_line:
        print("Không vẽ được vạch dừng. Thử lại.")
        continue

    traffic_lights_data[light_id] = {"roi": roi, "stop_line": stop_line}

    save_to_json(traffic_lights_data)

cap.release()
print("Hoàn tất.")
