# main.py
from train_modules import train_initial
from resume import resume_training
from logger import log_training

def log_checkpoint_status():
    """Kiểm tra trạng thái checkpoint."""
    import os
    checkpoint_path = "runs/detect/train_detect_plate/weights/last.pt"
    if os.path.exists(checkpoint_path):
        log_training(f"Checkpoint found at {checkpoint_path}. Resuming training.")
        return True
    else:
        log_training("No checkpoint found. Starting new training session.")
        return False

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Hỗ trợ trên Windows

    # Kiểm tra trạng thái checkpoint
    if log_checkpoint_status():
        # Nếu checkpoint tồn tại, tiếp tục huấn luyện
        resume_training()
        log_training("Training resumed.")
    else:
        # Nếu không có checkpoint, bắt đầu huấn luyện mới
        train_initial()
        log_training("Initial training completed.")
