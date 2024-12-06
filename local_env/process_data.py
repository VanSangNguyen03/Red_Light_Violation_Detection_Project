import os

labels_dir = "C:/Users/PC/Documents/professional_project/Red_Light_Violation_Detection_Project/data/train_traffic_light/labels"

for file in os.listdir(labels_dir):
    file_path = os.path.join(labels_dir, file)
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Lọc các dòng chỉ chứa dữ liệu hộp (boxes)
    filtered_lines = [line for line in lines if len(line.split()) == 5]

    # Ghi đè file nhãn chỉ với dữ liệu hộp
    with open(file_path, 'w') as f:
        f.writelines(filtered_lines)

print("Xử lý hoàn tất!")
