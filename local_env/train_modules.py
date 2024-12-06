# train.py
from ultralytics import YOLO
from logger import log_training  # Import hàm log

def train_initial():
    """Huấn luyện lần đầu và lưu checkpoint."""
    log_training("Starting initial training...")

    # Tải mô hình và huấn luyện
    model = YOLO("models/yolov8n.pt")  # Chọn phiên bản YOLO phù hợp
    results = model.train(
        data="configs/data_plate.yaml", 
        epochs=100, 
        batch=10, 
        amp=True, 
        device="0", 
        save=True,  # Đảm bảo lưu checkpoint
        project="runs/detect",  # Thư mục lưu log
        name="train_detect_plate"  # Tên cụ thể cho lần huấn luyện
    )
    
    log_training("Initial training completed.")
    print("Initial training completed.")
