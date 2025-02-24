import cv2
import numpy as np
import face_recognition
import easyocr
import os

# Inicializar EasyOCR para leer placas
reader = easyocr.Reader(['en'], gpu=False)  # 'en' para inglés; cambiar según idioma de placas

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
        known_face_names.append(filename.split(".")[0])  # Nombre sin extensión

# Configurar captura de video (0 para webcam, o URL RTSP para cámara IP)
cap = cv2.VideoCapture(0)  # Cambiar a "rtsp://user:pass@IP/cam" si usan cámara IP

# Variables para detección de movimiento
prev_frame = None

while True:
    # Leer frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar video.")
        break

    # Redimensionar para acelerar procesamiento
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

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
        # Dibujar rectángulo y nombre
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # --- Lectura de Placas (OCR) ---
    results = reader.readtext(small_frame)
    for (bbox, text, prob) in results:
        if prob > 0.5:  # Umbral de confianza
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0] * 2), int(top_left[1] * 2))
            bottom_right = (int(bottom_right[0] * 2), int(bottom_right[1] * 2))
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

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
        if cv2.contourArea(contour) > 500:  # Umbral de tamaño
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Movimiento", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    prev_frame = gray

    # Mostrar resultado
    cv2.imshow("SecureCam MVP", frame)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()