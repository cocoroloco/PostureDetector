import cv2
import mediapipe as mp
import math

# Inicializar MediaPipe para la detección de postura
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Función para calcular la distancia euclidiana entre dos puntos
def calcular_distancia(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Función para calcular el ángulo entre dos puntos
def calcular_angulo(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

# Capturar el video desde la cámara
cap = cv2.VideoCapture(0)

# Variables para almacenar valores de referencia
distancia_referencia = None
angulo_referencia = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a RGB (requerido por MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar keypoints
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        h, w, _ = frame.shape

        # Obtener coordenadas de puntos clave
        hombro_izq = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        hombro_der = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nariz = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        # Convertir coordenadas normalizadas a píxeles
        px_hombro_izq = (int(hombro_izq.x * w), int(hombro_izq.y * h))
        px_hombro_der = (int(hombro_der.x * w), int(hombro_der.y * h))
        px_nariz = (int(nariz.x * w), int(nariz.y * h))

        # Punto medio entre los hombros (como aproximación del pecho)
        px_pecho = ((px_hombro_izq[0] + px_hombro_der[0]) // 2, (px_hombro_izq[1] + px_hombro_der[1]) // 2)

        # Dibujar los puntos clave en el frame
        cv2.circle(frame, px_nariz, 5, (0, 255, 0), -1)
        cv2.circle(frame, px_pecho, 5, (0, 255, 0), -1)
        cv2.circle(frame, px_hombro_izq, 5, (255, 0, 0), -1)
        cv2.circle(frame, px_hombro_der, 5, (255, 0, 0), -1)

        # Calcular la distancia entre la nariz y el pecho
        distancia_cara_pecho = calcular_distancia(px_nariz, px_pecho)

        # Calcular el ángulo entre los hombros
        angulo_inclinacion = abs(calcular_angulo(px_hombro_izq, px_hombro_der))

        # Mostrar el ángulo y la distancia en pantalla
        cv2.putText(frame, f'Angulo: {int(angulo_inclinacion)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Distancia: {int(distancia_cara_pecho)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Si se han establecido valores de referencia
        if distancia_referencia is not None and angulo_referencia is not None:
            # Calcular umbrales relativos 
            umbral_angulo = angulo_referencia * 0.01
            umbral_distancia = distancia_referencia * 0.1

            # Verificar si la postura es correcta basándose en los valores relativos
            postura_correcta = True

            if abs(angulo_inclinacion - angulo_referencia) > umbral_angulo:
                cv2.putText(frame, 'Mala postura: Inclinacion excesiva!', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                postura_correcta = False

            if abs(distancia_cara_pecho - distancia_referencia) > umbral_distancia:
                cv2.putText(frame, 'Mala postura: Enderezate!', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                postura_correcta = False

            if postura_correcta:
                cv2.putText(frame, 'Buena postura', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Instrucciones para establecer la postura correcta
            cv2.putText(frame, 'Adopta buena postura y presiona "s"', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostrar el video
    cv2.imshow('Postura', frame)

    # Capturar la tecla presionada
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Guardar valores de referencia al presionar 's'
        distancia_referencia = distancia_cara_pecho
        angulo_referencia = angulo_inclinacion
        print("Valores de referencia guardados.")

cap.release()
cv2.destroyAllWindows()
