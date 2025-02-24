import cv2
import numpy as np
import face_recognition
import easyocr
import os

# Inicializar EasyOCR para leer placas
reader = easyocr.Reader(['en'], gpu=False)  # 'en' para inglés; ajusta según idioma de placas

# Cargar rostros conocidos
known_face_encodings = []
known_face_names = []
known_faces_dir = "known_faces"

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        base_name = filename.split(".")[0].split("_")[0]  # Toma solo la parte antes del "_"
        known_face_names.append(base_name)

# Seleccionar video (cambiar entre "street_video.mp4" o "office_video.mp4")
video_path = "office_video.mp4"  # Cambia a "office_video.mp4" para probar el otro
cap = cv2.VideoCapture(video_path)

# Verificar si el video se abrió
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Variables para detección de movimiento
prev_frame = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video.")
        break

    # Redimensionar para acelerar procesamiento (ajustado para videos reales)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame_count += 1

    # Procesar cada 5 frames para mejorar rendimiento en videos complejos
    if frame_count % 5 != 0:
        cv2.imshow("SecureCam - Procesando", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        continue

    # --- Reconocimiento Facial ---
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"Rostro: {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Lectura de Placas (OCR) ---
    results = reader.readtext(small_frame)
    for (bbox, text, prob) in results:
        if prob > 0.6 and len(text) > 2:  # Umbral más alto y longitud mínima para placas
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
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Umbral mayor para videos reales
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Movimiento Detectado", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    prev_frame = gray

    # Mostrar resultado
    cv2.imshow("SecureCam - Procesando", frame)

    # Salir con 'q' o pausar con 'p'
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(-1)  # Pausar hasta presionar otra tecla

# Liberar recursos
cap.release()
cv2.destroyAllWindows()