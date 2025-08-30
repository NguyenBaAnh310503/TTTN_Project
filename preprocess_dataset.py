import os
import cv2

DATASET_DIR = "dataset"       # dataset gốc (nền trắng – chữ đen)
PROCESSED_DIR = "processed_chars"    # dataset đã xử lý

os.makedirs(PROCESSED_DIR, exist_ok=True)

for folder in os.listdir(DATASET_DIR):
    src_folder = os.path.join(DATASET_DIR, folder)
    dst_folder = os.path.join(PROCESSED_DIR, folder)
    os.makedirs(dst_folder, exist_ok=True)
    
    for file in os.listdir(src_folder):
        img = cv2.imread(os.path.join(src_folder, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = cv2.bitwise_not(img)  # đảo màu: chữ trắng – nền đen
        cv2.imwrite(os.path.join(dst_folder, file), img)

print("✅ Hoàn tất xử lý ảnh, lưu tại:", PROCESSED_DIR)
