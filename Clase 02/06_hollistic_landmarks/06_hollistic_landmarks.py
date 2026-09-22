"""
===============================================================================
 06 - LANDMARKS HOLÍSTICOS (Holistic Landmarker)
===============================================================================

Este script usa la API moderna "MediaPipe Tasks" para detectar en tiempo
real los puntos clave del cuerpo, las manos y el rostro SIMULTÁNEAMENTE.

Holistic Landmarker combina tres modelos en uno:
  - Pose Landmarks:     33 puntos del cuerpo
  - Face Landmarks:     478 puntos del rostro
  - Hand Landmarks:     21 puntos por mano (42 en total)

IMPORTANTE: los tres conjuntos de puntos se calculan SIEMPRE juntos, en cada
frame, porque son parte de un único modelo. Los toggles de este script
(pose / rostro / manos) sólo deciden qué se DIBUJA en pantalla, no aceleran
ni desactivan el cálculo interno del modelo.

DESCARGA DEL MODELO
-------------------
https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker

Buscar la sección "Models" y descargar:
    holistic_landmarker.task
y colocarlo en la misma carpeta que este script.

CONTROLES (con la ventana de la cámara activa)
-----------------------------------------------
    q / ESC   -> cerrar el programa
    i         -> mostrar / ocultar el panel de información
    p         -> mostrar / ocultar landmarks de POSE
    r         -> mostrar / ocultar landmarks de ROSTRO
    m         -> mostrar / ocultar landmarks de MANOS
    ESPACIO   -> pausar / reanudar la captura
                 (al pausar, se congela el último frame y se analiza
                 ese frame con detalle: landmarks de pose, rostro,
                 mano izquierda y mano derecha con sus coordenadas)

NOTA: en pausa, el panel detallado de la derecha sólo muestra la
información de las capas que están visibles (activadas con p / r / m)
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

MODEL_FILENAME = "holistic_landmarker.task"
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME

# Confianza mínima (entre 0.0 y 1.0) para que el modelo dé por detectada
# cada parte del cuerpo.
#   - Valores más ALTOS  -> detecciones más "seguras", pero se pueden perder
#                           detecciones válidas (falsos negativos).
#   - Valores más BAJOS  -> detecta más fácil, pero puede haber falsos
#                           positivos (ruido).
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_HAND_LANDMARKS_CONFIDENCE = 0.5

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
DIBUJAR_POSE_INICIAL = True
DIBUJAR_ROSTRO_INICIAL = True
DIBUJAR_MANOS_INICIAL = True

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

# Cuántos landmarks mostrar por bloque en el panel de pausa.
# 478 es demasiado para mostrar de una, así que los cortamos en bloques.
# Si el panel se queda sin espacio, se muestra "..." y se corta.
MAX_LANDMARKS_ROSTRO_PAUSA = 60


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
    Dibuja sobre `frame` los landmarks detectados por el modelo, respetando
    qué capas están activadas en `estado` (diccionario con claves
    "pose", "rostro" y "manos").

    `resultado` es el objeto que devuelve `landmarker.detect_for_video(...)`.
    """
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_styles = mp.tasks.vision.drawing_styles

    # --- Pose (33 puntos del cuerpo) ---
    if estado["pose"] and resultado.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=resultado.pose_landmarks,
            connections=mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )

    # --- Manos (21 puntos por mano) ---
    if estado["manos"]:
        if resultado.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=resultado.left_hand_landmarks,
                connections=mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
            )
        if resultado.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=resultado.right_hand_landmarks,
                connections=mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
            )

    # --- Rostro (478 puntos) ---
    if estado["rostro"] and resultado.face_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=resultado.face_landmarks,
            connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(80, 80, 80), thickness=1
            ),
        )


# =============================================================================
# PANEL DE INFORMACIÓN
# =============================================================================

def dibujar_panel_info(frame, fps, estado):
    """
    Dibuja (si `estado["panel"]` es True) un panel semitransparente en la
    esquina superior izquierda con:
        - los FPS suavizados
        - el estado ON/OFF de cada capa (pose, rostro, manos)
        - el estado de pausa

    Si el panel está desactivado, esta función no dibuja nada y retorna
    inmediatamente.
    """
    if not estado["panel"]:
        return

    filas_toggle = [
        ("[p] Pose", estado["pose"]),
        ("[r] Rostro", estado["rostro"]),
        ("[m] Manos", estado["manos"]),
    ]

    # Geometría del panel
    pad = 12
    alto_fps = 22
    alto_linea = 20
    alto_panel = pad * 3 + alto_fps + 4 * alto_linea

    ancho_panel = ANCHO_PANEL_PAUSA if estado["pausa"] else ANCHO_PANEL

    alto_frame, ancho_frame, _ = frame.shape
    x1, y1 = 10, 10
    x2 = min(x1 + ancho_panel, ancho_frame - 10)
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
    y += alto_fps + pad

    # Estado de cada capa (ON en verde, OFF en gris)
    for nombre, activo in filas_toggle:
        color = COLOR_TEXTO_ON if activo else COLOR_TEXTO_OFF
        texto = f"{nombre}: {'ON' if activo else 'OFF'}"
        cv2.putText(
            frame,
            texto,
            (x1 + pad, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        y += alto_linea

    # Estado de pausa (se muestra siempre, al final del panel)
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

def _dibujar_bloque_landmarks(frame, x1, y, x2, y2, titulo, landmarks, max_mostrados=None):
    """
    Helper interno: dibuja un bloque de landmarks en el panel de pausa.

    Devuelve la coordenada `y` siguiente (después del bloque), o None si
    ya no hay espacio vertical y hay que cortar el panel.

    `max_mostrados` permite limitar cuántos landmarks del bloque se
    muestran (útil para el rostro, que tiene 478).
    """
    pad = 12
    cv2.putText(
        frame, titulo, (x1 + pad, y + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXTO, 1, cv2.LINE_AA,
    )
    y += 18

    cv2.putText(
        frame, "  #    x     y     z",
        (x1 + pad, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
    )
    y += 14

    total = len(landmarks)
    limite = total if max_mostrados is None else min(total, max_mostrados)

    for j in range(limite):
        if y + 12 > y2 - pad:
            cv2.putText(
                frame, "...", (x1 + pad, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
            )
            return None  # no hay más espacio

        lm = landmarks[j]
        texto = f"{j:3d}  {lm.x:.2f}  {lm.y:.2f}  {lm.z:.2f}"
        cv2.putText(
            frame, texto, (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 14

    if max_mostrados is not None and total > max_mostrados:
        cv2.putText(
            frame, f"... (+{total - max_mostrados} más)",
            (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 14

    return y + 6  # separación entre bloques


def dibujar_panel_pausa(frame, resultado, estado):
    """
    Cuando la captura está pausada, muestra en el lado derecho de la pantalla
    la información disponible del frame congelado, LIMITADA a las capas
    que están visibles según `estado`.

    Es decir:
        - Si `estado["pose"]` es True, muestra los 33 landmarks de pose.
        - Si `estado["rostro"]` es True, muestra los 478 landmarks del
          rostro (recortados a MAX_LANDMARKS_ROSTRO_PAUSA por espacio).
        - Si `estado["manos"]` es True, muestra los 21 landmarks de la
          mano izquierda y de la mano derecha.

    Si una capa está apagada (con p / r / m), su bloque NO se muestra.
    Esto tiene sentido pedagógico: el panel de pausa refleja lo que
    estás estudiando en ese momento.
    """
    alto_frame, ancho_frame, _ = frame.shape

    # Panel a la derecha
    pad = 12
    ancho_detalle = 320
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

    # --- Pose (sólo si la capa está activada) ---
    if estado["pose"] and resultado.pose_landmarks:
        nuevo_y = _dibujar_bloque_landmarks(
            frame, x1, y, x2, y2,
            "POSE (33 landmarks)",
            resultado.pose_landmarks,
        )
        if nuevo_y is None:
            return
        y = nuevo_y

    # --- Rostro (sólo si la capa está activada) ---
    if estado["rostro"] and resultado.face_landmarks:
        nuevo_y = _dibujar_bloque_landmarks(
            frame, x1, y, x2, y2,
            f"ROSTRO ({len(resultado.face_landmarks)} landmarks)",
            resultado.face_landmarks,
            max_mostrados=MAX_LANDMARKS_ROSTRO_PAUSA,
        )
        if nuevo_y is None:
            return
        y = nuevo_y

    # --- Manos (sólo si la capa está activada) ---
    if estado["manos"]:
        if resultado.left_hand_landmarks:
            nuevo_y = _dibujar_bloque_landmarks(
                frame, x1, y, x2, y2,
                "MANO IZQUIERDA (21 landmarks)",
                resultado.left_hand_landmarks,
            )
            if nuevo_y is None:
                return
            y = nuevo_y

        if resultado.right_hand_landmarks:
            nuevo_y = _dibujar_bloque_landmarks(
                frame, x1, y, x2, y2,
                "MANO DERECHA (21 landmarks)",
                resultado.right_hand_landmarks,
            )
            if nuevo_y is None:
                return
            y = nuevo_y


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
    elif tecla == ord("r"):
        estado["rostro"] = not estado["rostro"]
    elif tecla == ord("m"):
        estado["manos"] = not estado["manos"]
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
        print("https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker")
        print("y colocalo junto a este script.")
        return

    # Configuración del detector
    opciones = vision.HolisticLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_hand_landmarks_confidence=MIN_HAND_LANDMARKS_CONFIDENCE,
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
        "rostro": DIBUJAR_ROSTRO_INICIAL,
        "manos": DIBUJAR_MANOS_INICIAL,
        "pausa": False,
    }

    contador_fps = ContadorFPS(INTERVALO_ACTUALIZACION_FPS)

    print("Controles: [q]/[ESC] salir | [i] panel | [p] pose | [r] rostro | [m] manos | [ESPACIO] pausa")

    try:
        try:
            landmarker = vision.HolisticLandmarker.create_from_options(opciones)
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

                # Detección: siempre calcula pose + rostro + manos juntos
                resultado = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                # Dibujar sólo las capas activadas en estado_visualizacion
                dibujar_landmarks(frame, resultado, estado_visualizacion)

                # FPS suavizados + panel de información
                fps = contador_fps.actualizar()
                dibujar_panel_info(frame, fps, estado_visualizacion)

                # Si estamos en pausa, mostramos el panel detallado a la derecha
                # (respetando qué capas están visibles)
                if estado_visualizacion["pausa"]:
                    dibujar_panel_pausa(frame, resultado, estado_visualizacion)

                cv2.imshow(
                    "06 - Holistic Landmarks (q para salir)",
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