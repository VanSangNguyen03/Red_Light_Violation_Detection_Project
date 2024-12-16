
import cv2
import json

# Đọc dữ liệu ánh xạ từ file JSON
with open("json/traffic_lights1.json", "r") as file:
    traffic_lights_data = json.load(file)

# Mở video
video_path = "data/videos/sample2.mp4"  # Đường dẫn video của bạn
cap = cv2.VideoCapture(video_path)

# Kiểm tra nếu video mở thành công
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Đọc từng khung hình trong video và hiển thị kết quả
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Nếu không còn khung hình thì thoát

    # Vẽ ROI và vạch dừng cho mỗi đèn giao thông
    for light_id, data in traffic_lights_data.items():
        roi = data["roi"]
        stop_line = data["stop_line"]

        # Vẽ ROI (Hình chữ nhật)
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Vẽ vạch dừng (Dùng đường thẳng cho mỗi cặp điểm)
        for i in range(len(stop_line) - 1):
            cv2.line(frame, tuple(stop_line[i]), tuple(stop_line[i + 1]), (0, 0, 255), 2)

    # Hiển thị khung hình
    cv2.imshow("Video with Traffic Lights and Stop Lines", frame)

    # Nhấn 'q' để thoát khỏi video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
