from ultralytics import YOLO

# Tải mô hình đã huấn luyện (ví dụ: last.pt hoặc best.pt)
model = YOLO('runs/detect/train_detect_plate/weights/last.pt')  # Hoặc best.pt

# Cấu hình lại huấn luyện (nếu cần)
model.train(
    data='configs/data_plate.yaml',         # Đường dẫn đến tệp cấu hình dữ liệu
    epochs=50,                # Số epoch bạn muốn huấn luyện thêm
    batch=10,                 # Số lượng batch size
    imgsz=640,                # Kích thước ảnh đầu vào
    lr0=0.001,                 # Learning rate ban đầu
    lrf=0.2,                  # Learning rate final
    save_period=1,            # Lưu mô hình sau mỗi epoch
    weights='runs/detect/train_detect_plate/weights/last.pt'  # Tiếp tục huấn luyện từ mô hình này
)
