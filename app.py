import os
import cv2
import numpy as np
import uuid
from flask import Flask, render_template, request, url_for
import detect
import recognize

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result_file = None
    original_file = None
    plate_text = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "Chưa chọn file!"
        else:
            # Lưu ảnh gốc upload
            filename = f"upload_{uuid.uuid4().hex}.jpg"
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(upload_path)
            original_file = url_for('static', filename=f"uploads/{filename}")

            # đọc ảnh bằng OpenCV
            img = cv2.imread(upload_path)
            if img is None:
                error = "Không đọc được file ảnh."
            else:
                # phát hiện biển số bằng detect.py
                plate_img = detect.detect_plate(img)
                if plate_img is None:
                    error = "Không phát hiện được biển số!"
                else:
                    try:
                        annotated_img, plate_text = recognize.recognize_plate_image(plate_img)
                    except Exception as e:
                        error = "Lỗi khi nhận diện: " + str(e)
                        annotated_img = plate_img

                    # lưu ảnh kết quả
                    result_name = f"result_{uuid.uuid4().hex}.jpg"
                    result_path = os.path.join(app.config["RESULT_FOLDER"], result_name)
                    cv2.imwrite(result_path, annotated_img)
                    result_file = url_for('static', filename=f"results/{result_name}")

    return render_template("index.html",
                           result_file=result_file,
                           original_file=original_file,
                           plate_text=plate_text,
                           error=error)

if __name__ == "__main__":
    app.run(debug=True)
