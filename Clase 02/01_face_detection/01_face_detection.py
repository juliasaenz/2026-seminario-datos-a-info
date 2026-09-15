"""
================================================================================
 01 - DETECCIÓN DE ROSTROS (Face Detection)
================================================================================

Esta versión usa la API moderna "MediaPipe Tasks".

Por cada rostro que encuentra en la imagen, devuelve:

  - Una caja delimitadora (bounding box):
      x, y, ancho y alto del rostro.
  - Puntos clave (keypoints), como:
      ojo derecho, ojo izquierdo, punta de la nariz,
      centro de la boca, oreja derecha y oreja izquierda.

DESCARGA DEL MODELO
-------------------
El modelo se descarga desde el MediaPipe Model Zoo:
https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector

Buscar el bloque "Models" y descargar:
    blaze_face_short_range.tflite

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

# Archivo del modelo de detección facial
# (debe estar en la misma carpeta que este script)
MODEL_FILENAME = "blaze_face_short_range.tflite"
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME


# Confianza mínima para aceptar una detección.
# Un valor más alto:
#   - Reduce posibles falsos positivos
#   - Puede perder caras parcialmente ocultas o con poca iluminación
# Un valor más bajo:
#   - Detecta caras con mayor facilidad
#   - Puede producir más falsos positivos
MIN_DETECTION_CONFIDENCE = 0.5


# Índice de la cámara. (0 suele ser la webcam integrada)
# Si hay más de una cámara conectada, probar con 1, 2, etc.
CAMARA_INDEX = 0


# Resolución de captura. Una resolución más chica = más FPS
ANCHO_CAPTURA = 640
ALTO_CAPTURA = 480


# Nombres para los keypoints
NOMBRES_KEYPOINTS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
]


# ==============================================================================
# DIBUJO
# ==============================================================================

def dibujar_deteccion_personalizada(frame, deteccion):
    """
    Dibuja sobre el frame la información de una detección:
      - Un rectángulo alrededor del rostro.
      - Un círculo sobre cada punto clave, con su nombre.
      - Un porcentaje de confianza cerca de la caja del rostro.
    """

    alto, ancho, _ = frame.shape

    # --------------------------------------------------------------------------
    # Bounding Box
    # --------------------------------------------------------------------------
    # MediaPipe devuelve la caja en coordenadas de píxeles.
    # origin_x y origin_y indican la esquina superior izquierda.
    # width y height indican el tamaño de la caja.
    caja = deteccion.bounding_box

    # Recortamos a los bordes del frame para evitar dibujar fuera de pantalla.
    x = max(0, caja.origin_x)
    y = max(0, caja.origin_y)

    x2 = min(ancho - 1, caja.origin_x + caja.width)
    y2 = min(alto - 1, caja.origin_y + caja.height)

    cv2.rectangle(
        frame,
        (x, y),
        (x2, y2),
        color=(0, 255, 0),   # BGR: verde
        thickness=2,
    )

    # --------------------------------------------------------------------------
    # Keypoints
    # --------------------------------------------------------------------------
    # Los puntos clave tienen coordenadas normalizadas:
    #   x: valor entre 0.0 y 1.0 según el ancho de la imagen
    #   y: valor entre 0.0 y 1.0 según el alto de la imagen
    #
    # Multiplicamos x por el ancho e y por el alto para pasar a píxeles.
    for i, punto_clave in enumerate(deteccion.keypoints):
        x_px = int(punto_clave.x * ancho)
        y_px = int(punto_clave.y * alto)

        cv2.circle(
            frame,
            (x_px, y_px),
            radius=6,
            color=(0, 0, 255),  # BGR: rojo
            thickness=-1,       # -1 = círculo relleno
        )

        # Etiqueta del keypoint (si tenemos un nombre para ese índice).
        if i < len(NOMBRES_KEYPOINTS):
            cv2.putText(
                frame,
                NOMBRES_KEYPOINTS[i],
                (x_px + 8, y_px - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )

    # --------------------------------------------------------------------------
    # Confianza de la detección
    # --------------------------------------------------------------------------
    # La detección puede incluir varias categorías; para este modelo usamos la primera.
    if deteccion.categories:
        confianza = deteccion.categories[0].score
        texto = f"{confianza * 100:.0f}%"

        # Colocamos el texto arriba de la caja cuando hay espacio.
        posicion_texto_y = max(25, y - 10)

        cv2.putText(
            frame,
            texto,
            (x, posicion_texto_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),   # BGR: azul
            2,
        )


def dibujar_fps(frame, fps):
    """Muestra los FPS en la esquina superior izquierda."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),   # BGR: rojo
        2,
    )


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
        print("https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector")
        print("y colocalo junto a este script.")
        return

    # --------------------------------------------------------------------------
    # Configuración del detector MediaPipe Tasks
    # --------------------------------------------------------------------------
    # VIDEO indica que vamos a procesar una secuencia de imágenes
    opciones = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )

    # --------------------------------------------------------------------------
    # Abrir la cámara
    # --------------------------------------------------------------------------
    cap = cv2.VideoCapture(CAMARA_INDEX)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        print("Probá cambiar CAMARA_INDEX.")
        return

    # Fijamos la resolución: más chica = más rápido
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAPTURA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAPTURA)

    # Usamos try/finally para asegurarnos de liberar la cámara y cerrar las ventanas incluso si ocurre un error durante la ejecución.
    try:
        try:
            detector = vision.FaceDetector.create_from_options(opciones)
        except Exception as e:
            print(f"Error al crear el detector: {e}")
            print("Verificá que el archivo del modelo no esté corrupto.")
            return

        with detector:
            # Para calcular FPS medimos el tiempo entre frames
            tiempo_previo = time.time()

            while cap.isOpened():
                ok, frame = cap.read()

                if not ok:
                    print("No se pudo leer un frame de la cámara.")
                    break

                # Espejar horizontalmente la imagen
                frame = cv2.flip(frame, 1)

                # OpenCV lee las imágenes en formato BGR
                # MediaPipe espera imágenes en formato RGB
                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                # Convertimos el frame de NumPy a un objeto MediaPipe Image
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                # Cada frame recibe un timestamp estrictamente mayor que el del frame anterior
                timestamp_ms = int(time.time() * 1000)

                # Ejecutamos el detector sobre el frame actual
                resultado = detector.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                # Dibujamos cada rostro detectado
                for deteccion in resultado.detections:
                    dibujar_deteccion_personalizada(
                        frame,
                        deteccion,
                    )

                # Calculamos y dibujamos FPS
                tiempo_actual = time.time()
                delta = tiempo_actual - tiempo_previo
                fps = 1.0 / delta if delta > 0 else 0.0
                tiempo_previo = tiempo_actual

                dibujar_fps(frame, fps)

                # Mostramos el resultado
                cv2.imshow(
                    "01 - Deteccion de Rostros (q para salir)",
                    frame,
                )

                # Esperamos 1 milisegundo por una tecla
                tecla = cv2.waitKey(1) & 0xFF

                # q = salir
                # 27 = ESC
                if tecla == ord("q") or tecla == 27:
                    break

    finally:
        # Liberamos la cámara y cerramos todas las ventanas de OpenCV
        cap.release()
        # Un waitKey extra ayuda a que algunos backends (GTK/Qt) cierren bien
        cv2.waitKey(1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()