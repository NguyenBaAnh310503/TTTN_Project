import cv2
import numpy as np
import os

def preprocess_image(image):
    """Chuyển ảnh sang grayscale và làm mượt ảnh"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def detect_plate(image):
    """Phát hiện biển số, trả về ảnh cắt biển số hoặc None nếu không tìm thấy"""
    preprocessed = preprocess_image(image)
    edged = cv2.Canny(preprocessed, 100, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    plate_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is None:
        return None

    mask = np.zeros(preprocessed.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)

    x, y = np.where(mask == 255)
    if len(x) == 0 or len(y) == 0:
        return None

    topx, topy = np.min(x), np.min(y)
    bottomx, bottomy = np.max(x), np.max(y)
    cropped = image[topx:bottomx+1, topy:bottomy+1]
    return cropped


def process_images(input_dir, output_dir):
    """Xử lý tất cả ảnh trong thư mục đầu vào và lưu kết quả vào thư mục đầu ra"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:

            continue

        plate_img = detect_plate(img)
        if plate_img is not None:
            output_path = os.path.join(output_dir, f"plate_{img_name}")
            cv2.imwrite(output_path, plate_img)
        


if __name__ == "__main__":
    input_folder = 'Input_images'
    output_folder = 'Output_plates'
    process_images(input_folder, output_folder)
