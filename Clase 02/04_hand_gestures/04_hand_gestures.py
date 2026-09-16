"""
================================================================================
 04 - RECONOCIMIENTO DE GESTOS (Hand Gesture Recognition)
================================================================================

Esta versión usa la API moderna "MediaPipe Tasks" para reconocer gestos
de la mano en tiempo real.

El modelo detecta la mano, encuentra sus 21 puntos clave, y además
clasifica QUÉ gesto está haciendo. Los gestos predefinidos son:
  - None: Sin gesto reconocido
  - Closed_Fist: Puño cerrado
  - Open_Palm: Palma abierta
  - Pointing_Up: Dedo índice apuntando hacia arriba
  - Thumb_Down: Pulgar hacia abajo
  - Thumb_Up: Pulgar hacia arriba
  - Victory: Seña de victoria (dos dedos)
  - ILoveYou: Seña "te amo" (lengua de señas)

DESCARGA DEL MODELO
-------------------
https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer

Buscar la sección "Models" y descargar:
    gesture_recognizer.task

CONTROLES
---------
- Tocar 'q' o ESC para cerrar la ventana.
================================================================================
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==============================================================================
# PARÁMETROS EDITABLES
# ==============================================================================

MODEL_FILENAME = "gesture_recognizer.task"
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME


# Confianza mínima para detectar una mano.
MIN_HAND_DETECTION_CONFIDENCE = 0.5

# Confianza mínima para considerar que hay una mano presente.
MIN_HAND_PRESENCE_CONFIDENCE = 0.5

# Confianza mínima para el seguimiento entre frames.
MIN_TRACKING_CONFIDENCE = 0.5

CAMARA_INDEX = 0

ANCHO_CAPTURA = 640
ALTO_CAPTURA = 480

# Cantidad máxima de manos a detectar.
NUM_HANDS = 4

# Colores (BGR) del panel de información
COLOR_PANEL = (30, 30, 30)
COLOR_TEXTO = (240, 240, 240)
COLOR_TEXTO_TENUE = (170, 170, 170)

ANCHO_PANEL = 260
ALPHA_PANEL = 0.65

# ==============================================================================
# DIBUJO
# ==============================================================================

def dibujar_landmarks(frame, resultado):
    """
    Dibuja los puntos de la mano y sus conexiones.

    Usa las utilidades de dibujo de MediaPipe Tasks.
    """
    if not resultado.hand_landmarks:
        return

    mp_drawing = mp.tasks.vision.drawing_utils
    mp_styles = mp.tasks.vision.drawing_styles

    for landmarks in resultado.hand_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
        )


def dibujar_panel_info(frame, fps, resultado):
    """Panel semitransparente con FPS y gestos reconocidos."""
    # Información a mostrar
    info = []
    if resultado.gestures:
        for i, gestos in enumerate(resultado.gestures):
            if gestos:
                gesto = gestos[0]
                nombre = gesto.category_name
                score = gesto.score
                # Traducir el nombre del gesto al español
                traduccion = {
                    "None": "Sin gesto",
                    "Closed_Fist": "Puño",
                    "Open_Palm": "Palma abierta",
                    "Pointing_Up": "Apuntando",
                    "Thumb_Down": "Pulgar abajo",
                    "Thumb_Up": "Pulgar arriba",
                    "Victory": "Victoria",
                    "ILoveYou": "Te amo",
                }
                etiqueta = traduccion.get(nombre, nombre)
                info.append(f"Mano {i + 1}: {etiqueta} ({score:.2f})")

    if not info:
        info.append("Sin manos")

    # Geometría del panel
    pad = 12
    alto_panel = pad * 3 + 28 + len(info) * 20
    alto_frame, ancho_frame, _ = frame.shape
    x1, y1 = 10, 10
    x2 = min(x1 + ANCHO_PANEL, ancho_frame - 10)
    y2 = min(y1 + alto_panel, alto_frame - 10)

    # Fondo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, ALPHA_PANEL, frame, 1 - ALPHA_PANEL, 0, frame)

    # FPS
    y = y1 + pad
    cv2.putText(
        frame,
        f"FPS {fps:5.1f}",
        (x1 + pad, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_TEXTO,
        1,
        cv2.LINE_AA,
    )

    # Filas de información
    y += 28 + pad
    for linea in info:
        cv2.putText(
            frame,
            linea,
            (x1 + pad, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_TEXTO_TENUE,
            1,
            cv2.LINE_AA,
        )
        y += 20


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    if not MODEL_PATH.exists():
        print("No se encontró el archivo del modelo:")
        print(MODEL_PATH)
        print()
        print(f"Descargá '{MODEL_FILENAME}' desde:")
        print("https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer")
        print("y colocalo junto a este script.")
        return

    # Configuración del detector
    opciones = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(CAMARA_INDEX)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        print("Probá cambiar CAMARA_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAPTURA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAPTURA)

    try:
        try:
            recognizer = vision.GestureRecognizer.create_from_options(opciones)
        except Exception as e:
            print(f"Error al crear el detector: {e}")
            print("Verificá que el archivo del modelo no esté corrupto.")
            return

        with recognizer:
            tiempo_previo = time.time()

            while cap.isOpened():
                ok, frame = cap.read()

                if not ok:
                    print("No se pudo leer un frame de la cámara.")
                    break

                # Espejar horizontalmente
                frame = cv2.flip(frame, 1)

                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # NumPy -> MediaPipe Image
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                # Timestamp real
                timestamp_ms = int(time.time() * 1000)

                # Detectar
                resultado = recognizer.recognize_for_video(
                    mp_image,
                    timestamp_ms,
                )

                # Dibujar
                dibujar_landmarks(frame, resultado)

                # FPS
                tiempo_actual = time.time()
                delta = tiempo_actual - tiempo_previo
                fps = 1.0 / delta if delta > 0 else 0.0
                tiempo_previo = tiempo_actual

                # Panel
                dibujar_panel_info(frame, fps, resultado)

                # Mostrar
                cv2.imshow(
                    "04 - Gestos de la Mano (q para salir)",
                    frame,
                )

                tecla = cv2.waitKey(1) & 0xFF

                if tecla == ord("q") or tecla == 27:
                    break

    finally:
        cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()