import unittest
import os
import cv2

# import hàm detect_plate từ file detect.py
from detect import detect_plate  

INPUT_DIR = "Input_images"

class TestDetectPlate(unittest.TestCase):

    def test_all_images(self):
        """Kiểm thử toàn bộ ảnh trong thư mục Input_images"""
        images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in images:
            with self.subTest(img=img_name):
                img_path = os.path.join(INPUT_DIR, img_name)
                img = cv2.imread(img_path)

                self.assertIsNotNone(img, f"Không đọc được ảnh {img_name}")

                result = detect_plate(img)

                # Test: detect_plate phải trả về kết quả khác None
                self.assertIsNotNone(result, f"Không phát hiện được biển số trong {img_name}")

if __name__ == "__main__":
    unittest.main()
