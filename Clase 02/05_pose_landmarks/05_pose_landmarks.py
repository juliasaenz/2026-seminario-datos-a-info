"""
===============================================================================
 05 - PUNTOS DE LA POSE (Pose Landmarks)
===============================================================================

Este script usa la API moderna "MediaPipe Tasks" para detectar puntos clave
del cuerpo humano en tiempo real.

Pose Landmarker detecta 33 puntos clave en 3D que cubren:
  - Rostro: nariz, ojos, orejas, boca
  - Torso: hombros, caderas
  - Brazos: codos, muñecas, manos
  - Piernas: rodillas, tobillos, pies

Además devuelve coordenadas en dos sistemas:
  - Coordenadas normalizadas de imagen (0.0 a 1.0)
  - Coordenadas de mundo (metros aproximados, relativo a la cadera)

DESCARGA DEL MODELO
-------------------
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

Buscar la sección "Models" y descargar:
    pose_landmarker_lite.task

CONTROLES (con la ventana de la cámara activa)
-----------------------------------------------
    q / ESC   -> cerrar el programa
    i         -> mostrar / ocultar el panel de información
    p         -> mostrar / ocultar los landmarks de la pose
    ESPACIO   -> pausar / reanudar la captura
                 (al pausar, se congela el último frame y se analiza
                 ese frame con detalle: los 33 landmarks con sus
                 coordenadas x, y, z, visibility y presence)
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
# Todo lo que afecta cómo MediaPipe detecta

MODEL_FILENAME = "pose_landmarker_lite.task"
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME

# Confianza mínima (entre 0.0 y 1.0) para que el modelo dé por detectada una pose
MIN_POSE_DETECTION_CONFIDENCE = 0.5

# Confianza mínima para considerar que la pose sigue "presente" una vez detectada
# (se usa frame a frame, no sólo en la detección inicial)
MIN_POSE_PRESENCE_CONFIDENCE = 0.5

# Confianza mínima para el seguimiento (tracking) de los puntos entre frames
#   Valores más bajos = tracking más "suelto" (puede perderse más fácil);
#   valores más altos = más estricto (puede tardar más en reengancharse)
MIN_TRACKING_CONFIDENCE = 0.5

# Cantidad máxima de personas (poses) a detectar simultáneamente
NUM_POSES = 1

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

# Estado inicial al arrancar el programa
MOSTRAR_PANEL_INICIAL = True
DIBUJAR_POSE_INICIAL = True

# Cada cuántos segundos se recalcula el número de FPS
INTERVALO_ACTUALIZACION_FPS = 0.5

# Colores en formato BGR (el que usa OpenCV, no RGB).
COLOR_PANEL = (30, 30, 30)
COLOR_TEXTO = (240, 240, 240)
COLOR_TEXTO_TENUE = (170, 170, 170)
COLOR_TEXTO_ON = (120, 220, 120)     # verde suave: capa activada
COLOR_TEXTO_OFF = (110, 110, 110)    # gris: capa desactivada
COLOR_TEXTO_PAUSA = (120, 180, 255)  # naranja suave: modo pausa

ANCHO_PANEL = 230        # ancho normal
ANCHO_PANEL_PAUSA = 320  # ancho cuando está pausado (más info)
ALPHA_PANEL = 0.65       # transparencia del fondo (0 = invisible, 1 = opaco)


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
# DIBUJO DE LANDMARKS
# =============================================================================

def dibujar_landmarks(frame, resultado, estado):
    """
    Dibuja sobre `frame` los landmarks de pose detectados (puede haber más
    de una persona si NUM_POSES > 1), respetando el toggle `estado["pose"]`.

    `resultado` es el objeto que devuelve `landmarker.detect_for_video(...)`.
    """
    if not estado["pose"] or not resultado.pose_landmarks:
        return

    mp_drawing = mp.tasks.vision.drawing_utils
    mp_styles = mp.tasks.vision.drawing_styles

    for landmarks in resultado.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )


# =============================================================================
# PANEL DE INFORMACIÓN
# =============================================================================

def dibujar_panel_info(frame, fps, resultado, estado):
    """
    Dibuja (si `estado["panel"]` es True) un panel semitransparente en la
    esquina superior izquierda con:
        - los FPS suavizados
        - el estado ON/OFF del dibujo de la pose
        - el estado de pausa (si está pausado, se indica claramente)
        - la cantidad de poses (personas) detectadas en el frame actual

    Si el panel está desactivado, esta función no dibuja nada y retorna
    inmediatamente.
    """
    if not estado["panel"]:
        return

    cantidad_poses = len(resultado.pose_landmarks) if resultado.pose_landmarks else 0

    # ----------------------------------------------------------------------
    # Geometría del panel
    # ----------------------------------------------------------------------
    pad = 12
    alto_fps = 28
    alto_linea = 20
    alto_panel = pad * 3 + alto_fps + 3 * alto_linea  # 3 filas: toggle, poses, pausa

    ancho_panel = ANCHO_PANEL_PAUSA if estado["pausa"] else ANCHO_PANEL

    alto_frame, ancho_frame, _ = frame.shape
    x1, y1 = 10, 10
    x2 = min(x1 + ancho_panel, ancho_frame - 10)
    y2 = min(y1 + alto_panel, alto_frame - 10)

    # ----------------------------------------------------------------------
    # Fondo semitransparente
    # ----------------------------------------------------------------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, ALPHA_PANEL, frame, 1 - ALPHA_PANEL, 0, frame)

    # ----------------------------------------------------------------------
    # FPS
    # ----------------------------------------------------------------------
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
    y += alto_fps + pad

    # ----------------------------------------------------------------------
    # Estado del toggle de pose (ON en verde, OFF en gris)
    # ----------------------------------------------------------------------
    color_estado = COLOR_TEXTO_ON if estado["pose"] else COLOR_TEXTO_OFF
    texto_estado = f"[p] Pose: {'ON' if estado['pose'] else 'OFF'}"
    cv2.putText(
        frame, texto_estado, (x1 + pad, y + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_estado, 1, cv2.LINE_AA,
    )
    y += alto_linea

    # ----------------------------------------------------------------------
    # Cantidad de poses (personas) detectadas
    # ----------------------------------------------------------------------
    cv2.putText(
        frame, f"Poses detectadas: {cantidad_poses}", (x1 + pad, y + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
    )
    y += alto_linea

    # ----------------------------------------------------------------------
    # Estado de pausa (se muestra siempre, al final del panel)
    # ----------------------------------------------------------------------
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
# PANEL DETALLADO EN PAUSA
# =============================================================================

def dibujar_panel_pausa(frame, resultado):
    """
    Cuando la captura está pausada, muestra en el lado derecho de la pantalla
    toda la información disponible de cada pose detectada:

        - Los 33 landmarks con sus coordenadas (x, y, z)
        - Los valores de visibility y presence por landmark

    Es mucha información, por eso se muestra sólo en pausa y en un panel
    aparte, para no saturar la vista durante la captura en vivo.
    """
    if not resultado.pose_landmarks:
        return

    alto_frame, ancho_frame, _ = frame.shape

    # Panel a la derecha
    pad = 12
    ancho_detalle = 300
    x1 = ancho_frame - ancho_detalle - 10
    y1 = 10
    x2 = ancho_frame - 10
    y2 = alto_frame - 10

    # Fondo semitransparente más oscuro para diferenciar del panel de info
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    y = y1 + pad

    # Título
    cv2.putText(
        frame, "DETALLE (en pausa)", (x1 + pad, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO, 1, cv2.LINE_AA,
    )
    y += 30

    # Por cada pose
    for i, landmarks in enumerate(resultado.pose_landmarks):
        # Encabezado de la pose
        cv2.putText(
            frame, f"Pose {i + 1}", (x1 + pad, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXTO, 1, cv2.LINE_AA,
        )
        y += 20

        # Encabezado de columnas
        cv2.putText(
            frame, "  #   x     y     z    vis   pres",
            (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 16

        # Los 33 landmarks
        for j, lm in enumerate(landmarks):
            # Si se acaba el espacio vertical, cortamos
            if y + 12 > y2 - pad:
                cv2.putText(
                    frame, "...", (x1 + pad, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
                )
                return

            # Algunos landmarks pueden no tener visibility / presence
            vis = getattr(lm, "visibility", None)
            pres = getattr(lm, "presence", None)
            vis_txt = f"{vis:.2f}" if vis is not None else " -- "
            pres_txt = f"{pres:.2f}" if pres is not None else " -- "

            texto = f"{j:2d}  {lm.x:.2f}  {lm.y:.2f}  {lm.z:.2f}  {vis_txt}  {pres_txt}"
            cv2.putText(
                frame, texto, (x1 + pad, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
            )
            y += 14

        # Separación entre poses
        y += 10


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
    elif tecla == ord("p"):
        estado["pose"] = not estado["pose"]
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
        print("https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker")
        print("y colocalo junto a este script.")
        return

    # Configuración del detector
    opciones = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
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
        "pose": DIBUJAR_POSE_INICIAL,
        "pausa": False,
    }

    contador_fps = ContadorFPS(INTERVALO_ACTUALIZACION_FPS)

    print("Controles: [q]/[ESC] salir | [i] panel | [p] pose | [ESPACIO] pausa")

    try:
        try:
            landmarker = vision.PoseLandmarker.create_from_options(opciones)
        except Exception as e:
            print(f"Error al crear el detector: {e}")
            print("Verificá que el archivo del modelo no esté corrupto.")
            return

        with landmarker:
            # Frame congelado cuando está en pausa.
            # Si es None, estamos en vivo.
            frame_congelado = None

            while cap.isOpened():
                # ------------------------------------------------------------------
                # ¿De dónde sacamos el frame?
                # ------------------------------------------------------------------
                if estado_visualizacion["pausa"] and frame_congelado is not None:
                    # En pausa: usamos el frame congelado y NO leemos de la cámara
                    frame = frame_congelado.copy()
                else:
                    # En vivo: leemos un frame nuevo de la cámara
                    ok, frame = cap.read()
                    if not ok:
                        print("No se pudo leer un frame de la cámara.")
                        break

                    # Espejar horizontalmente para que se sienta como un espejo
                    frame = cv2.flip(frame, 1)

                    # Si entramos recién en pausa, guardamos este frame
                    if estado_visualizacion["pausa"]:
                        frame_congelado = frame.copy()

                # BGR (OpenCV) -> RGB (MediaPipe)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # NumPy array -> objeto mp.Image que espera el modelo
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                # Timestamp real en milisegundos, requerido en modo VIDEO
                # (en pausa seguimos usando timestamps crecientes;
                #  MediaPipe no permite repetir el mismo timestamp)
                timestamp_ms = int(time.time() * 1000)

                # Detección de pose(s)
                resultado = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                # Dibujar sólo si el toggle de pose está activado
                dibujar_landmarks(frame, resultado, estado_visualizacion)

                # FPS suavizados + panel de información
                fps = contador_fps.actualizar()
                dibujar_panel_info(frame, fps, resultado, estado_visualizacion)

                # Si estamos en pausa, mostramos el panel detallado a la derecha
                if estado_visualizacion["pausa"]:
                    dibujar_panel_pausa(frame, resultado)

                cv2.imshow(
                    "05 - Puntos de la Pose (q para salir)",
                    frame,
                )

                tecla = cv2.waitKey(1) & 0xFF

                # Detectar transición de pausa -> reanudar y limpiar el frame congelado
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