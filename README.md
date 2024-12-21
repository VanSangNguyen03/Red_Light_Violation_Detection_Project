# Traffic Violation Vehicle Detection System

## Project Description
This project develops an intelligent traffic monitoring system, using modern AI technologies such
as YOLOv8 and PaddleOCR to automatically detect vehicles violating traffic laws, especially red light
violations. The system is capable of recognizing vehicles, license plates, traffic light status and 
storing violation information effectively.

---

## Main functions
1. **Vehicle Detection:**
- Identify vehicle types (motorbikes, cars, trucks) in real-time video.

- Track vehicles by unique ID.

2. **Vehicle License Plate Recognition:**
- Use PaddleOCR to extract and recognize characters from license plates.

- Works well in low light conditions or blurred license plates.

3. **Traffic light status detection:**
- Use YOLOv8 to identify traffic light status (red, green, yellow).

- Identify red light violations based on light status and vehicle location.

4. **Violation data management:**
- Store violation information (license plate, time, image) in the database.

- Support retrieval and export of violation reports.

5. **User interface:**
- Provide an intuitive web interface to manage violation information.
- Support searching, editing, deleting and exporting reports.

---

## Technology used

### 1. **YOLOv8**
- **Purpose:** Detect vehicles and traffic light status.

- **Features:**
- Real-time processing with high speed and accuracy.

- Well integrated with complex traffic systems.

### 2. **PaddleOCR**
- **Purpose:** License plate recognition.
- **Features:**
- Good support for Vietnamese language.
- Effective in low light conditions.

### 3. **OpenCV**
- **Purpose:** Video and image processing.
- **Features:**
- Analyze each frame from the input video.
- Integrate processing pipeline for AI models.

### 4. **Database:**
- **SQL:** Store structured violation information (license plate, time, status).
- **JSON:** Store unstructured data such as violation images/videos.

### 5. **NVIDIA GPU**
- **Purpose:** Accelerate model training and processing.
- **Features:**
- Supports CUDA and cuDNN to optimize performance.

### 6. **Frameworks and other tools:**
- **Flask/Streamlit:** Web interface development.
- **Conda:** Environment and library management.
- **Visual Studio Code:** Code editing and debugging.

---

## Install and run the project

### 1. **System requirements:**
- Python >= 3.8
- NVIDIA GPU with CUDA support (recommended)
- Required libraries (included in `requirements.txt`)

### 2. **Install:**
```bash
# Clone repository
$ git clone <repository-url>

# Go to project directory
$ cd traffic-violation-system

# Create virtual environment
$ conda create -n traffic-env python=3.8
$ conda activate traffic-env

# Install libraries
$ pip install -r requirements.txt
```

### 3. **Run the system:**
```bash
# Run web interface
$ run main_tkinter.py
```

---

## Project structure
```
Red_Light_Violation_Detection_Project/
│
├── configs/
│   ├── yolov8_plate_detection.yaml
│   └── yolov8_traffic_light.yaml
│
├── data/
│   ├── license_plates/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   └── traffic_lights/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
│
├── json/
│   ├── traffic_lights1.json
│   └── camera_configs.json
│
├── models/
│   ├── yolov8n.pt
│   ├── plate_detection_best.pt
│   └── traffic_light_best.pt
│
├── results/
│   └── violations/
│       ├── images/
│       │   └── violation_id{ID}_{timestamp}.jpg
│       └── violating_vehicles.json
│
├── runs/
│   └── detect/
│       ├── train_detect_plate/
│       │   └── weights/
│       │       └── last.pt
│       └── train_traffic_light_status/
│           └── weights/
│               └── last.pt
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
│
├── src/
│   ├── main_tkinter.py
│   └── sort.py
│
├── modules/
│   ├── detection.py
│   ├── tracking.py
│   ├── plate_recognition.py
│   └── database.py
│
├── training_log/
│   ├── plate_detection_metrics.csv
│   └── traffic_light_metrics.csv
│
├── ViolatingDatabase.db
│
└── requirements.txt
```

---

## Demo
- **Illustrative image:**
- Real-time video processing with bounding box around vehicle and license plate.
![image](https://github.com/user-attachments/assets/85fbdc5d-3ead-4431-ba86-03b2da03375e)

- Web interface displaying list of violations.
![main interface image](https://github.com/user-attachments/assets/0ad44b0f-f5fa-42cf-89d7-0144845028db)

- **Demo video:** https://youtu.be/PSx6haXlIHs

---

## Author
- **Nguyen Van Sang** - Full-stack AI Developer.
- **Contact:** [Email](mailto:vansang.nguyen21503@gmail.com)

---

## Development direction
1. Expand the recognition of other violations such as wrong lane driving, illegal parking.
2. Improve real-time processing performance in crowded traffic areas.
3. Integrate the system with the cloud platform to expand processing and storage capabilities.
4. Develop a mobile application to support data management anytime, anywhere.

---

## Contribution
- Any comments or bug reports, please send them to the Issues section on GitHub or contact us directly via email.
