import sqlite3
#kết nối đến cơ sở dữ liệu (HOẶC tạo mới nếu chưa tồn tại)
conn = sqlite3.connect('ViolatingDatabase.db')
#tạo con trỏ để thực thi các câu lệnh SQL
cursor = conn.cursor()
#tạo một bảng License Plates trong cơ sở dữ liệu
cursor.execute(
    '''
    CREATE TABLE IF NOT EXISTS LicensePlates(
        STT INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_id TEXT NOT NULL,
        license_plate TEXT NOT NULL,
        vehicle_class TEXT,
        time_stamp DATETIME,
        status TEXT,
        image_path TEXT
    )
    '''
    )
