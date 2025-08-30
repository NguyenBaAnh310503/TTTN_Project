import unittest
import cv2
import os
import recognize  # module nhận diện biển số

class TestRecognizePlate(unittest.TestCase):
    def test_recognize(self):
        # Thư mục chứa các biển số đã tách
        folder = "Output_plates"

        # Lấy danh sách file ảnh
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        self.assertGreater(len(images), 0, "Không có ảnh nào trong Output_plates để test")

        for img_name in images:
            img_path = os.path.join(folder, img_name)
            plate_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            self.assertIsNotNone(plate_img, f"Không đọc được ảnh {img_name}")

            # Gọi hàm nhận diện
            text = recognize.recognize_plate_image(plate_img)

            print(f"Ảnh: {img_name} -> Kết quả nhận diện: {text}")

            # Kiểm tra output (ít nhất phải khác rỗng)
            self.assertTrue(text is not None and text != "",
                            f"Không nhận diện được biển số từ ảnh {img_name}")

if __name__ == "__main__":
    unittest.main()
