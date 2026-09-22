"""
===============================================================================
 01 - DETECCIÓN DE ROSTROS (Face Detection)
===============================================================================

Este script usa la API moderna "MediaPipe Tasks" para detectar rostros
en tiempo real.

Face Detector devuelve, por cada rostro encontrado:
  - Una caja delimitadora (bounding box): origin_x, origin_y, width, height.
  - 6 keypoints con coordenadas normalizadas (x, y): ojo derecho, ojo
    izquierdo, punta de la nariz, centro de la boca, oreja derecha y
    oreja izquierda.
  - Un score de confianza (0.0 a 1.0) de que eso sea un rostro.

DESCARGA DEL MODELO
-------------------
https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector

Buscar la sección "Models" y descargar:
    blaze_face_short_range.tflite
y colocarlo en la misma carpeta que este script.

CONTROLES (con la ventana de la cámara activa)
-----------------------------------------------
    q / ESC   -> cerrar el programa
    i         -> mostrar / ocultar el panel de información
    d         -> mostrar / ocultar el dibujo de detecciones
    ESPACIO   -> pausar / reanudar la captura
                 (al pausar, se congela el último frame y se analiza
                 ese frame con detalle: por cada rostro, la caja con
                 sus valores crudos y los 6 keypoints con sus
                 coordenadas normalizadas. Además, cada keypoint se
                 numera sobre la imagen para cruzarlo con el panel)
===============================================================================
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =============================================================================
# 1) PARÁMETROS EDITABLES DEL MODELO
# =============================================================================
# Todo lo que afecta cómo MediaPipe detecta (no cómo se ve en pantalla)

MODEL_FILENAME = "blaze_face_short_range.tflite"
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME

# Confianza mínima (entre 0.0 y 1.0) para aceptar una detección.
#   Valores más ALTOS  -> menos falsos positivos, pero se pueden perder
#                         caras parcialmente ocultas o con poca luz.
#   Valores más BAJOS  -> detecta más fácil, pero puede haber más ruido.
MIN_DETECTION_CONFIDENCE = 0.5

# Índice de la cámara a usar. Si hay más de una cámara conectada,
# probar 0, 1, 2... hasta encontrar la correcta
CAMARA_INDEX = 0

# Resolución de captura pedida a la cámara (la cámara puede no respetarla
# exactamente; depende del driver / hardware)
ANCHO_CAPTURA = 640
ALTO_CAPTURA = 480


# =============================================================================
# 2) PARÁMETROS EDITABLES DEL PANEL Y LA VISUALIZACIÓN
# =============================================================================
# Todo lo relacionado a QUÉ se dibuja y CÓMO se ve el panel de información

# Estado inicial al arrancar el programa. Durante la ejecución se pueden
# togglear con el teclado
MOSTRAR_PANEL_INICIAL = True
DIBUJAR_DETECCIONES_INICIAL = True

# Cada cuántos segundos se recalcula el número de FPS
INTERVALO_ACTUALIZACION_FPS = 0.5

# Colores en formato BGR (el que usa OpenCV, no RGB).
COLOR_PANEL = (30, 30, 30)
COLOR_TEXTO = (240, 240, 240)
COLOR_TEXTO_TENUE = (170, 170, 170)
COLOR_TEXTO_ON = (120, 220, 120)     # verde suave: capa activada
COLOR_TEXTO_OFF = (110, 110, 110)    # gris: capa desactivada
COLOR_TEXTO_PAUSA = (120, 180, 255)  # naranja suave: modo pausa

# Colores de los elementos de la detección
COLOR_CAJA = (0, 255, 0)       # verde para la caja
COLOR_KEYPOINT = (0, 0, 255)   # rojo para los keypoints

ANCHO_PANEL = 240          # ancho del panel de info
ANCHO_PANEL_PAUSA = 320    # ancho cuando está pausado (más info)
ANCHO_PANEL_DERECHA = 340  # ancho del panel de la derecha (detalle en pausa)
ALPHA_PANEL = 0.65         # transparencia del fondo (0 = invisible, 1 = opaco)


# =============================================================================
# CONTADOR DE FPS SUAVIZADO
# =============================================================================

class ContadorFPS:
    """
    Calcula un valor de FPS "suavizado".

    En vez de mostrar el FPS instantáneo de cada frame (que salta mucho de
    un frame a otro), este contador acumula frames durante una ventana de
    tiempo (`intervalo_actualizacion`) y sólo recalcula el número mostrado
    cuando esa ventana se completa. El resultado es un número mucho más
    fácil de leer.
    """

    def __init__(self, intervalo_actualizacion: float = 0.5):
        self.intervalo_actualizacion = intervalo_actualizacion
        self._ultimo_tiempo = time.time()
        self._contador_frames = 0
        self._fps_mostrado = 0.0

    def actualizar(self) -> float:
        """Llamar una vez por cada frame procesado. Devuelve el FPS a mostrar."""
        self._contador_frames += 1
        ahora = time.time()
        transcurrido = ahora - self._ultimo_tiempo

        if transcurrido >= self.intervalo_actualizacion:
            self._fps_mostrado = self._contador_frames / transcurrido
            self._contador_frames = 0
            self._ultimo_tiempo = ahora

        return self._fps_mostrado


# =============================================================================
# DIBUJO DE DETECCIONES
# =============================================================================

def dibujar_detecciones(frame, resultado, estado):
    """
    Dibuja sobre `frame` cada detección de rostro:
      - La caja delimitadora (verde).
      - Los keypoints como círculos rellenos (rojos).

    Si `estado["pausa"]` es True, además dibuja el índice numérico al lado
    de cada keypoint, para poder cruzarlo con el panel de detalle de la
    derecha.
    """
    if not estado["detecciones"] or not resultado.detections:
        return

    alto, ancho, _ = frame.shape

    for deteccion in resultado.detections:
        # --------------------------------------------------------------------
        # Bounding Box
        # --------------------------------------------------------------------
        caja = deteccion.bounding_box
        x = max(0, caja.origin_x)
        y = max(0, caja.origin_y)
        x2 = min(ancho - 1, caja.origin_x + caja.width)
        y2 = min(alto - 1, caja.origin_y + caja.height)

        cv2.rectangle(
            frame, (x, y), (x2, y2),
            color=COLOR_CAJA, thickness=2,
        )

        # --------------------------------------------------------------------
        # Keypoints
        # --------------------------------------------------------------------
        for j, punto_clave in enumerate(deteccion.keypoints):
            x_px = int(punto_clave.x * ancho)
            y_px = int(punto_clave.y * alto)

            cv2.circle(
                frame, (x_px, y_px),
                radius=6,
                color=COLOR_KEYPOINT,
                thickness=-1,   # -1 = círculo relleno
            )

            # En pausa, numeramos cada keypoint para poder cruzarlo con
            # el panel de detalle de la derecha.
            if estado["pausa"]:
                cv2.putText(
                    frame, f"{j}",
                    (x_px + 8, y_px - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR_KEYPOINT, 1, cv2.LINE_AA,
                )


# =============================================================================
# PANEL DE INFORMACIÓN BÁSICO (izquierda)
# =============================================================================

def dibujar_panel_info(frame, fps, resultado, estado):
    """
    Dibuja (si `estado["panel"]` es True) un panel semitransparente en la
    esquina superior izquierda con:
        - los FPS suavizados
        - el estado ON/OFF del dibujo de detecciones
        - la cantidad de rostros detectados
        - la confianza de cada rostro detectado
        - el estado de pausa
    """
    if not estado["panel"]:
        return

    filas_toggle = [
        ("[d] Detecciones", estado["detecciones"]),
    ]

    # Confianza de cada detección
    n_rostros = len(resultado.detections) if resultado.detections else 0
    filas_confianza = []
    if resultado.detections:
        for i, deteccion in enumerate(resultado.detections):
            if deteccion.categories:
                score = deteccion.categories[0].score
                filas_confianza.append(f"Rostro {i + 1}: {score:.3f}")

    if not filas_confianza:
        filas_confianza.append("Sin rostros")

    # Geometría del panel
    pad = 12
    alto_fps = 22
    alto_linea = 20
    n_filas = len(filas_toggle) + 1 + len(filas_confianza) + 1
    alto_panel = pad * 3 + alto_fps + n_filas * alto_linea

    ancho_panel = ANCHO_PANEL_PAUSA if estado["pausa"] else ANCHO_PANEL

    alto_frame, ancho_frame, _ = frame.shape
    x1, y1 = 10, 10
    x2 = min(x1 + ancho_panel, ancho_frame - 10)
    y2 = min(y1 + alto_panel, alto_frame - 10)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, ALPHA_PANEL, frame, 1 - ALPHA_PANEL, 0, frame)

    # FPS
    y = y1 + pad
    cv2.putText(
        frame, f"FPS {fps:5.1f}", (x1 + pad, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXTO, 1, cv2.LINE_AA,
    )
    y += alto_fps + pad

    # Toggles
    for nombre, activo in filas_toggle:
        color = COLOR_TEXTO_ON if activo else COLOR_TEXTO_OFF
        texto = f"{nombre}: {'ON' if activo else 'OFF'}"
        cv2.putText(
            frame, texto, (x1 + pad, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )
        y += alto_linea

    # Conteo de rostros
    cv2.putText(
        frame, f"Rostros: {n_rostros}", (x1 + pad, y + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
    )
    y += alto_linea

    # Confianza por rostro
    for linea in filas_confianza:
        cv2.putText(
            frame, linea, (x1 + pad, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += alto_linea

    # Estado de pausa
    if estado["pausa"]:
        cv2.putText(
            frame, "[ESPACIO] EN PAUSA", (x1 + pad, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXTO_PAUSA, 1, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame, "[ESPACIO] en vivo", (x1 + pad, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXTO_OFF, 1, cv2.LINE_AA,
        )


# =============================================================================
# PANEL DETALLADO EN PAUSA (derecha)
# =============================================================================

def _geometria_panel_derecha(frame):
    """Devuelve (x1, y1, x2, y2) del panel de la derecha."""
    alto_frame, ancho_frame, _ = frame.shape
    x1 = ancho_frame - ANCHO_PANEL_DERECHA - 10
    y1 = 10
    x2 = ancho_frame - 10
    y2 = alto_frame - 10
    return x1, y1, x2, y2


def _dibujar_fondo_panel_derecha(frame, x1, y1, x2, y2):
    """Dibuja el fondo semitransparente del panel de la derecha."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)


def dibujar_panel_pausa(frame, resultado, estado):
    """
    En pausa, muestra el panel de la derecha con la información CRUDA que
    devuelve el modelo, tal cual, sin traducciones ni etiquetas inventadas:

        - Por cada detección:
            - score (confianza) tal como viene en `categories[0].score`
            - bounding_box: origin_x, origin_y, width, height
            - keypoints: índice + (x, y) normalizados, sin nombres

    Los índices de los keypoints coinciden con los números que se dibujan
    sobre la imagen cuando está en pausa.

    Si el panel se queda sin espacio, corta con "...".
    """
    if not estado["pausa"]:
        return

    if not resultado.detections:
        return

    x1, y1, x2, y2 = _geometria_panel_derecha(frame)
    _dibujar_fondo_panel_derecha(frame, x1, y1, x2, y2)

    pad = 12
    y = y1 + pad

    # Título
    cv2.putText(
        frame, "DETALLE (en pausa)", (x1 + pad, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO, 1, cv2.LINE_AA,
    )
    y += 28

    # Por cada detección
    for i, deteccion in enumerate(resultado.detections):
        if y + 12 > y2 - pad:
            cv2.putText(
                frame, "...", (x1 + pad, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
            )
            return

        # Encabezado
        cv2.putText(
            frame, f"deteccion[{i}]", (x1 + pad, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXTO, 1, cv2.LINE_AA,
        )
        y += 18

        # Score crudo
        if deteccion.categories:
            score = deteccion.categories[0].score
            cv2.putText(
                frame, f"  categories[0].score: {score:.4f}",
                (x1 + pad, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
            )
            y += 12

        # Bounding box cruda
        caja = deteccion.bounding_box
        cv2.putText(
            frame, "  bounding_box:", (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 12
        cv2.putText(
            frame, f"    origin_x: {caja.origin_x}",
            (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 12
        cv2.putText(
            frame, f"    origin_y: {caja.origin_y}",
            (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 12
        cv2.putText(
            frame, f"    width:    {caja.width}",
            (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 12
        cv2.putText(
            frame, f"    height:   {caja.height}",
            (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 14

        # Keypoints crudos
        cv2.putText(
            frame, "  keypoints:", (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 12

        for j, kp in enumerate(deteccion.keypoints):
            if y + 12 > y2 - pad:
                cv2.putText(
                    frame, "  ...", (x1 + pad, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
                )
                return

            texto = f"    [{j}] x={kp.x:.4f}  y={kp.y:.4f}"
            cv2.putText(
                frame, texto, (x1 + pad, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
            )
            y += 12

        # Separación entre detecciones
        y += 8


# =============================================================================
# MANEJO DE TECLADO
# =============================================================================

def procesar_tecla(tecla, estado) -> bool:
    """
    Interpreta la tecla presionada y actualiza `estado` (los toggles).

    Devuelve False si el programa debe cerrarse (q / ESC), True en caso
    contrario.
    """
    if tecla == ord("q") or tecla == 27:  # 27 = ESC
        return False

    if tecla == ord("i"):
        estado["panel"] = not estado["panel"]
    elif tecla == ord("d"):
        estado["detecciones"] = not estado["detecciones"]
    elif tecla == ord(" "):
        estado["pausa"] = not estado["pausa"]

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not MODEL_PATH.exists():
        print("No se encontró el archivo del modelo:")
        print(MODEL_PATH)
        print()
        print(f"Descargá '{MODEL_FILENAME}' desde:")
        print("https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector")
        print("y colocalo junto a este script.")
        return

    # Configuración del detector
    opciones = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )

    cap = cv2.VideoCapture(CAMARA_INDEX)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        print("Probá cambiar CAMARA_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAPTURA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAPTURA)

    # Estado inicial de los toggles
    estado_visualizacion = {
        "panel": MOSTRAR_PANEL_INICIAL,
        "detecciones": DIBUJAR_DETECCIONES_INICIAL,
        "pausa": False,
    }

    contador_fps = ContadorFPS(INTERVALO_ACTUALIZACION_FPS)

    print("Controles: [q]/[ESC] salir | [i] panel | [d] detecciones | [ESPACIO] pausa")

    try:
        try:
            detector = vision.FaceDetector.create_from_options(opciones)
        except Exception as e:
            print(f"Error al crear el detector: {e}")
            print("Verificá que el archivo del modelo no esté corrupto.")
            return

        with detector:
            # Frame congelado cuando está en pausa.
            frame_congelado = None

            # Contador de timestamps
            timestamp_ms = 0

            while cap.isOpened():
                # -------------------------------------------------------------
                # ¿De dónde sacamos el frame?
                # -------------------------------------------------------------
                if estado_visualizacion["pausa"] and frame_congelado is not None:
                    frame = frame_congelado.copy()
                else:
                    ok, frame = cap.read()
                    if not ok:
                        print("No se pudo leer un frame de la cámara.")
                        break
                    frame = cv2.flip(frame, 1)
                    if estado_visualizacion["pausa"]:
                        frame_congelado = frame.copy()

                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # NumPy -> MediaPipe Image
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                # Timestamp monotónicamente creciente (requisito de MediaPipe
                # en modo VIDEO).
                timestamp_ms += 1

                # Detectar
                resultado = detector.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                # Dibujar sólo si el toggle está activado
                dibujar_detecciones(frame, resultado, estado_visualizacion)

                # FPS
                fps = contador_fps.actualizar()

                # Panel de info (izquierda)
                dibujar_panel_info(frame, fps, resultado, estado_visualizacion)

                # Panel de pausa (derecha, sólo si está pausado)
                dibujar_panel_pausa(frame, resultado, estado_visualizacion)

                cv2.imshow(
                    "01 - Detección de Rostros (q para salir)",
                    frame,
                )

                tecla = cv2.waitKey(1) & 0xFF

                estaba_pausado = estado_visualizacion["pausa"]
                if not procesar_tecla(tecla, estado_visualizacion):
                    break
                if estaba_pausado and not estado_visualizacion["pausa"]:
                    frame_congelado = None

    finally:
        cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()