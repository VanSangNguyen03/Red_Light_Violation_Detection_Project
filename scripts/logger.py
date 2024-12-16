# logger.py
import logging

# Cấu hình logging
logging.basicConfig(
    filename="training_log.txt",  # Đường dẫn tệp log
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def log_training(message):
    """Ghi log thông báo."""
    logging.info(message)
