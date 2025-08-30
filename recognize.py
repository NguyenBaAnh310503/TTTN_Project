# recognize.py
import cv2
import numpy as np
import tensorflow as tf
import os

# model will be loaded lazily on first use
_model = None
_id_to_label = None

MODEL_PATH = "cnn_license_char_recognition.h5"
LABELS_PATH = "labels.npy"
IMG_SIZE = 28

def _load_model_and_labels():
    global _model, _id_to_label
    if _model is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            raise FileNotFoundError("Model or labels not found. Run train first.")
        _model = tf.keras.models.load_model(MODEL_PATH)
        label_to_id = np.load(LABELS_PATH, allow_pickle=True).item()
        _id_to_label = {v: k for k, v in label_to_id.items()}

def rotate_with_hough(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if lines is not None:
        angles = []
        for rho, theta in lines[:,0]:
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:
                angles.append(angle)
        if len(angles) > 0:
            median_angle = np.median(angles)
            if abs(median_angle) > 1.5:
                (h, w) = img.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
                return rotated
    return img

def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh

def segment_characters(thresh_img):
    # trả về list ảnh ký tự (28x28) và danh sách box tương ứng [(x,y,w,h), ...] sắp theo x tăng
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    boxes = []
    pad = 2
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 12 and w > 3:  # threshold nhỏ để giữ ký tự nhỏ
            x_start = max(x - pad, 0)
            y_start = max(y - pad, 0)
            x_end = min(x + w + pad, thresh_img.shape[1])
            y_end = min(y + h + pad, thresh_img.shape[0])
            char_img = thresh_img[y_start:y_end, x_start:x_end]
            char_img = cv2.resize(char_img, (IMG_SIZE, IMG_SIZE))
            chars.append(char_img)
            boxes.append((x, y, w, h))
    # sort theo x
    paired = sorted(zip(boxes, chars), key=lambda b: b[0][0])
    if not paired:
        return [], []
    boxes_sorted, chars_sorted = zip(*paired)
    return list(chars_sorted), list(boxes_sorted)

def recognize_characters(chars):
    _load_model_and_labels()
    texts = []
    for char in chars:
        # char is grayscale 28x28
        arr = char.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
        pred = _model.predict(arr, verbose=0)
        idx = int(np.argmax(pred))
        texts.append(_id_to_label.get(idx, '?'))
    return texts

def recognize_plate_image(plate_img):
    """
    Input: plate_img (BGR numpy array, cropped plate)
    Output: (annotated_img (BGR), plate_text (string concatenated))
    """
    _load_model_and_labels()
    rotated = rotate_with_hough(plate_img)
    thresh = preprocess_plate(rotated)
    chars, boxes = segment_characters(thresh)
    if not chars:
        # nothing recognized: return original crop and empty string
        return rotated, ""

    preds = recognize_characters(chars)
    # vẽ box và text trên ảnh rotated
    annotated = rotated.copy()
    for (x, y, w, h), label in zip(boxes, preds):
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = x + (w - text_w) // 2
        text_y = y + (h + text_h) // 2
        cv2.putText(annotated, label, (text_x, text_y), font, font_scale, (0,255,255), thickness, cv2.LINE_AA)

    plate_text = "".join(preds)
    return annotated, plate_text
