"""
================================================================================
 02 - PUNTOS FACIALES (Face Landmarks)
================================================================================

Esta versión usa la API moderna "MediaPipe Tasks" para detectar puntos
faciales en tiempo real

Face Landmarks devuelve:
  - Una malla completa de 478 puntos en 3D sobre el rostro.
  - 52 "blendshapes": coeficientes que describen la expresión facial
    (sonrisa, ceño fruncido, boca abierta, etc.).
  - Opcionalmente: una matriz de transformación facial para efectos 3D.

DESCARGA DEL MODELO
-------------------
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

Buscar la sección "Models" y descargar:
    face_landmarker.task

NOTA SOBRE LOS BLENDSHAPES
--------------------------
Los blendshapes son 52 coeficientes (0.0 a 1.0) que representan
expresiones faciales. Por ejemplo:
  - "mouthSmileLeft" / "mouthSmileRight" (sonrisa)
  - "eyeBlinkLeft" / "eyeBlinkRight" (parpadeo)
  - "jawOpen" (boca abierta)
  - "browDownLeft" / "browDownRight" (ceño fruncido)

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

# Archivo del modelo de puntos faciales
MODEL_FILENAME = "face_landmarker.task"
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME

# Confianza mínima para detectar un rostro
# Un valor más alto reduce falsos positivos
MIN_FACE_DETECTION_CONFIDENCE = 0.5

# Confianza mínima para considerar que hay un rostro presente
MIN_FACE_PRESENCE_CONFIDENCE = 0.5

# Confianza mínima para el seguimiento entre frames
MIN_TRACKING_CONFIDENCE = 0.5

# Índice de la cámara
CAMARA_INDEX = 0

# Resolución de captura
ANCHO_CAPTURA = 640
ALTO_CAPTURA = 480

# Cantidad máxima de rostros a detectar
# Con más de 1, el rendimiento baja
NUM_FACES = 1

# Qué dibujar sobre la malla facial
# Se pueden activar/desactivar para experimentar
DIBUJAR_MALLA = True       # Triangulación completa
DIBUJAR_CONTORNOS = False   # Contornos de ojos, cejas, labios, cara
DIBUJAR_IRIS = True        # Puntos del iris

# ==============================================================================
# DIBUJO
# ==============================================================================

def dibujar_landmarks(frame, resultado):
    """
    Dibuja la malla facial y los contornos sobre el frame
    """
    if not resultado.face_landmarks:
        return

    # Utilidades de dibujo de MediaPipe Tasks
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_styles = mp.tasks.vision.drawing_styles

    for i, landmarks in enumerate(resultado.face_landmarks):
        # --------------------------------------------------------------------
        # Malla completa (triangulación)
        # --------------------------------------------------------------------
        if DIBUJAR_MALLA:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(80, 80, 80), thickness=1
                ),
            )

        # --------------------------------------------------------------------
        # Contornos (ojos, cejas, labios, óvalo facial)
        # --------------------------------------------------------------------
        if DIBUJAR_CONTORNOS:
            # Óvalo facial
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                ),
            )
            # Cejas
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                ),
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                ),
            )
            # Ojos
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                ),
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                ),
            )
            # Labios
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                ),
            )

        # ---------------------------------------------------------------------
        # Iris
        # ---------------------------------------------------------------------
        if DIBUJAR_IRIS:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2
                ),
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2
                ),
            )

def dibujar_panel_info(frame, fps, resultado):
    
    # Colores (BGR) del panel de información
    COLOR_PANEL = (30, 30, 30)          # gris oscuro
    COLOR_TEXTO = (240, 240, 240)       # casi blanco
    COLOR_TEXTO_TENUE = (170, 170, 170) # gris claro

    ANCHO_PANEL = 240
    ALPHA_PANEL = 0.65

    etiquetas = [
    ("mouthSmileLeft",  "Sonrisa Izq"),
    ("mouthSmileRight", "Sonrisa Der"),
    ("eyeBlinkLeft",    "Parpadeo Izq"),
    ("eyeBlinkRight",   "Parpadeo Der"),
    ("jawOpen",         "Boca Abierta"),
    ("browDownLeft",    "Ceja Izq Baja"),
    ("browDownRight",   "Ceja Der Baja"),
    ("browInnerUp",     "Ceja Centro"),
    ("mouthPucker",     "Boca Fruncida"),
]

    valores = {}
    if resultado.face_blendshapes:
        for bs in resultado.face_blendshapes[0]:
            valores[bs.category_name] = bs.score

    pad = 12
    alto_panel = pad * 3 + 28 + len(etiquetas) * 20
    alto_frame, ancho_frame, _ = frame.shape
    x1, y1 = 10, 10
    x2 = min(x1 + ANCHO_PANEL, ancho_frame - 10)
    y2 = min(y1 + alto_panel, alto_frame - 10)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, ALPHA_PANEL, frame, 1 - ALPHA_PANEL, 0, frame)

    y = y1 + pad
    cv2.putText(frame, f"FPS {fps:5.1f}", (x1 + pad, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXTO, 1, cv2.LINE_AA)

    y += 28 + pad
    for nombre_interno, etiqueta in etiquetas:
        valor = valores.get(nombre_interno)
        texto = f"{etiqueta}: {valor:.2f}" if valor is not None else f"{etiqueta}: --"
        cv2.putText(frame, texto, (x1 + pad, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA)
        y += 20

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # --------------------------------------------------------------------------
    # Verificación del archivo del modelo
    # --------------------------------------------------------------------------
    if not MODEL_PATH.exists():
        print("No se encontró el archivo del modelo:")
        print(MODEL_PATH)
        print()
        print(f"Descargá '{MODEL_FILENAME}' desde:")
        print("https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker")
        print("y colocalo junto a este script.")
        return

    # --------------------------------------------------------------------------
    # Configuración del detector
    # --------------------------------------------------------------------------
    opciones = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=NUM_FACES,
        min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
        min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        output_face_blendshapes=True,  # Queremos las expresiones
    )

    # --------------------------------------------------------------------------
    # Abrir la cámara
    # --------------------------------------------------------------------------
    cap = cv2.VideoCapture(CAMARA_INDEX)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        print("Probá cambiar CAMARA_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAPTURA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAPTURA)

    try:
        try:
            landmarker = vision.FaceLandmarker.create_from_options(opciones)
        except Exception as e:
            print(f"Error al crear el detector: {e}")
            print("Verificá que el archivo del modelo no esté corrupto.")
            return

        with landmarker:
            tiempo_previo = time.time()

            while cap.isOpened():
                ok, frame = cap.read()

                if not ok:
                    print("No se pudo leer un frame de la cámara.")
                    break

                # Espejar horizontalmente
                frame = cv2.flip(frame, 1)

                # BGR -> RGB
                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                # NumPy -> MediaPipe Image
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                # Timestamp real
                timestamp_ms = int(time.time() * 1000)

                # Detectar
                resultado = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                # Dibujar malla y contornos
                dibujar_landmarks(frame, resultado)


                # FPS
                tiempo_actual = time.time()
                delta = tiempo_actual - tiempo_previo
                fps = 1.0 / delta if delta > 0 else 0.0
                tiempo_previo = tiempo_actual
                
                dibujar_panel_info(frame, fps, resultado)

                # Mostrar
                cv2.imshow(
                    "02 - Puntos Faciales (q para salir)",
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