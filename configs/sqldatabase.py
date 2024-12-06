import sqlite3
#kết nối đến cơ sở dữ liệu (HOẶC tạo mới nếu chưa tồn tại)
conn = sqlite3.connect('licensePlatesDatabase.db')
#tạo con trỏ để thực thi các câu lệnh SQL
cursor = conn.cursor()
#tạo một bảng License Plates trong cơ sở dữ liệu
cursor.execute(
    '''
    CREATE TABLE IF NOT EXISTS LicensePlates(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        license_plate TEXT NOT NULL
    )
    '''
    )
