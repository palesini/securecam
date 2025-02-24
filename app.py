import cv2
import numpy as np
import os
from flask import Flask, render_template, request, send_from_directory, url_for

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para subir y procesar video
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video file uploaded", 400
    video = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
    video.save(video_path)

    # Procesar el video
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_video.mp4')
    process_video(video_path, output_path)

    # Mostrar resultado
    video_url = url_for('static', filename='output_video.mp4')
    return render_template('index.html', video_url=video_url)

# Función para procesar el video (solo movimiento)
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    prev_frame = None
    frame_count = 0
    motion_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame_count += 1
        process_this_frame = frame_count % 5 == 0

        if process_this_frame:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if prev_frame is None:
                prev_frame = gray
                continue
            frame_diff = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_diff, 40, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > 10000:
                    motion_detected = True
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Movimiento", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if motion_detected:
                motion_count += 1
            else:
                motion_count = max(0, motion_count - 1)
            prev_frame = gray

        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)