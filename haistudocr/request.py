import requests
import os

# Dapatkan direktori saat ini
current_dir = os.getcwd()
print("Current directory:", current_dir)

# Path gambar - pastikan ini sesuai dengan lokasi file Anda
image_path = os.path.join(current_dir, './static/input_1.jpg')
print("Looking for image at:", image_path)

# Periksa apakah file ada
if not os.path.exists(image_path):
    print(f"Error: File tidak ditemukan di {image_path}")
    exit()

# URL API
url = 'http://localhost:5000/upload'

try:
    # Buka dan kirim file
    files = {'file': open(image_path, 'rb')}
    response = requests.post(url, files=files)
    
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)
    
except FileNotFoundError:
    print(f"File tidak ditemukan di {image_path}")
except Exception as e:
    print("Error:", str(e))