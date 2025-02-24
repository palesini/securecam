import cv2
import numpy as np
import face_recognition
import easyocr
import os

# Inicializar EasyOCR para leer placas
reader = easyocr.Reader(['en'], gpu=False)

# Cargar rostros conocidos con depuración
known_face_encodings = []
known_face_names = []
known_faces_dir = "known_faces"

print("Cargando rostros conocidos...")
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpeg"):
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

# Cargar el video
video_path = "office_video.mp4"  # Cambia a tu video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Variables para movimiento y persistencia de rostros
prev_frame = None
frame_count = 0
motion_count = 0
face_detections = []  # Lista para almacenar detecciones recientes

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video.")
        break

    # Reducir resolución para velocidad
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame_count += 1

    # Procesar cada 5 frames para balancear velocidad y detección
    process_this_frame = frame_count % 5 == 0

    # --- Mostrar Rostros Persistentes ---
    for detection in face_detections[:]:  # Copia para modificar mientras iteramos
        top, right, bottom, left, name, lifespan = detection
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"Rostro: {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        detection[5] -= 1  # Reducir vida útil
        if detection[5] <= 0:
            face_detections.remove(detection)

    if process_this_frame:
        # --- Reconocimiento Facial ---
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=1)  # Menos upsampling
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
            # Agregar detección con vida útil de 10 frames
            face_detections.append([top, right, bottom, left, name, 10])

        # --- Lectura de Placas (OCR) ---
        results = reader.readtext(small_frame)
        for (bbox, text, prob) in results:
            if prob > 0.7 and len(text) > 2:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = (int(top_left[0] * 2), int(top_left[1] * 2))
                bottom_right = (int(bottom_right[0] * 2), int(bottom_right[1] * 2))
                cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
                cv2.putText(frame, f"Placa: {text}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # --- Detección de Movimiento ---
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

    # Mostrar resultado
    cv2.imshow("SecureCam MVP", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()