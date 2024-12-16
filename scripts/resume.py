# resume.py
from ultralytics import YOLO
from logger import log_training  # Import hàm log


def resume_training():
    """Tiếp tục huấn luyện từ checkpoint."""
    log_training("Resuming training from checkpoint...")

    # Tải mô hình từ checkpoint và tiếp tục huấn luyện
    model = YOLO("runs/detect/train_detect_plate2/weights/last.pt")  # Tải checkpoint
    results = model.train(
        resume=True,  # Tiếp tục từ checkpoint
        epochs=120,  # Tổng số epoch mới (bao gồm epoch trước đó)
        batch=10, 
        amp=True, 
        device="0"
    )
    
    log_training("Training resumed and completed.")
    print("Training resumed.")
