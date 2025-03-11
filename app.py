import cv2
import numpy as np
import face_recognition
import os
from flask import Flask, render_template, request, send_from_directory, url_for

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cargar rostros conocidos
known_face_encodings = []
known_face_names = []
known_faces_dir = "known_faces"

print("Cargando rostros conocidos...")
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append(filename.split(".")[0].split("_")[0])
            print(f"Cargado: {filename}")
        else:
            print(f"Error: No se detectó rostro en {filename}")

if not known_face_encodings:
    print("Advertencia: No se cargaron rostros conocidos.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video file uploaded", 400
    video = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
    video.save(video_path)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_video.mp4')
    process_video(video_path, output_path)

    video_url = url_for('static', filename='output_video.mp4')
    return render_template('index.html', video_url=video_url)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    prev_frame = None
    frame_count = 0
    motion_count = 0
    face_detections = []  # Para persistencia de rostros

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame_count += 1
        process_this_frame = frame_count % 5 == 0

        # Mostrar rostros persistentes
        for detection in face_detections[:]:
            top, right, bottom, left, name, lifespan = detection
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"Rostro: {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            detection[5] -= 1
            if detection[5] <= 0:
                face_detections.remove(detection)

        if process_this_frame:
            # Reconocimiento facial
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=1)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Desconocido"
                if known_face_encodings:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                face_detections.append([top, right, bottom, left, name, 10])

            # Detección de movimiento
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