import os
from ultralytics import YOLO
import cv2

VIDEOS_DIR = os.path.join('.', 'data','videos')
video_path = os.path.join(VIDEOS_DIR, 'sample2.mp4')
if not os.path.isfile(video_path):
    print(f"Video không tồn tại tại đường dẫn: {video_path}")
else:
    print(f"Video đã được tìm thấy: {video_path}")

video_path_out = '{}_out.mp4'.format(video_path)

# Mở video
cap = cv2.VideoCapture(video_path)

# Kiểm tra xem video có mở thành công không
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Đọc khung hình đầu tiên
ret, frame = cap.read()

# Kiểm tra nếu khung hình không được đọc
if not ret:
    print("Không thể đọc khung hình đầu tiên.")
    exit()

# Lấy kích thước khung hình
H, W, _ = frame.shape

# Tạo video output
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'train', 'my_experiment12', 'weights', 'detect_plate_ex.pt')

# Load mô hình YOLO
model = YOLO(model_path)

threshold = 0.5

# Tiến hành xử lý video
while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Ghi khung hình vào video output
    out.write(frame)

    # Đọc khung hình tiếp theo
    ret, frame = cap.read()

    # Kiểm tra nếu khung hình tiếp theo không có
    if not ret:
        print("Đã đến cuối video.")
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()