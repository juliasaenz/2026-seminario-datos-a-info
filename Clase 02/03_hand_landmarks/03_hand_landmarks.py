"""
===============================================================================
 03 - PUNTOS DE LA MANO (Hand Landmarks)
===============================================================================

Este script usa la API moderna "MediaPipe Tasks" para detectar puntos clave
de las manos en tiempo real.

Hand Landmarker devuelve:
  - 21 puntos clave (landmarks) en 3D por cada mano detectada.
  - La lateralidad de la mano: izquierda o derecha.
  - Opcionalmente: coordenadas en espacio "world" (metros, relativo).

LOS 21 PUNTOS CLAVE
-------------------
MediaPipe numera los puntos del 0 al 20:

  0:     Muñeca (wrist)
  1-4:   Pulgar (thumb) - de la base a la punta
  5-8:   Índice (index) - de la base a la punta
  9-12:  Medio (middle) - de la base a la punta
  13-16: Anular (ring) - de la base a la punta
  17-20: Meñique (pinky) - de la base a la punta

NOTA SOBRE LOS DATOS POR PUNTO
------------------------------
Cada landmark de la mano expone x, y, z. A diferencia del modelo de pose,
Hand Landmarker NO devuelve visibility ni presence por punto, así que
esos son todos los datos disponibles por landmark individual.

DESCARGA DEL MODELO
-------------------
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

Buscar la sección "Models" y descargar:
    hand_landmarker.task
y colocarlo en la misma carpeta que este script.

CONTROLES (con la ventana de la cámara activa)
-----------------------------------------------
    q / ESC   -> cerrar el programa
    i         -> mostrar / ocultar el panel de información
    h         -> mostrar / ocultar los landmarks de la mano
    ESPACIO   -> pausar / reanudar la captura
                 (al pausar, se congela el último frame y se analiza
                 ese frame con detalle: lateralidad de cada mano y los
                 21 landmarks con sus coordenadas x, y, z)
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

MODEL_FILENAME = "hand_landmarker.task"
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME

# Confianza mínima (entre 0.0 y 1.0) para que el modelo dé por detectada una mano
MIN_HAND_DETECTION_CONFIDENCE = 0.5

# Confianza mínima para considerar que la mano sigue "presente" una vez detectada
# (se usa frame a frame, no sólo en la detección inicial)
MIN_HAND_PRESENCE_CONFIDENCE = 0.5

# Confianza mínima para el seguimiento (tracking) de los puntos entre frames
#   Valores más bajos = tracking más "suelto" (puede perderse más fácil);
#   valores más altos = más estricto (puede tardar más en reengancharse)
MIN_TRACKING_CONFIDENCE = 0.5

# Cantidad máxima de manos a detectar simultáneamente (entre 1 y 10)
NUM_HANDS = 2

# Índice de la cámara a usar. Si hay más de una cámara conectada,
# probar 0, 1, 2... hasta encontrar la correcta
CAMARA_INDEX = 0

# Resolución de captura pedida a la cámara (la cámara puede no respetarla
# exactamente; depende del driver / hardware)
ANCHO_CAPTURA = 1080
ALTO_CAPTURA = 720


# =============================================================================
# 2) PARÁMETROS EDITABLES DEL PANEL Y LA VISUALIZACIÓN
# =============================================================================
# Todo lo relacionado a QUÉ se dibuja y CÓMO se ve el panel de información

# Estado inicial al arrancar el programa. Durante la ejecución se pueden
# togglear con el teclado
MOSTRAR_PANEL_INICIAL = True
DIBUJAR_MANO_INICIAL = True

# Cada cuántos segundos se recalcula el número de FPS
INTERVALO_ACTUALIZACION_FPS = 0.5

# Colores en formato BGR (el que usa OpenCV, no RGB).
COLOR_PANEL = (30, 30, 30)
COLOR_TEXTO = (240, 240, 240)
COLOR_TEXTO_TENUE = (170, 170, 170)
COLOR_TEXTO_ON = (120, 220, 120)     # verde suave: capa activada
COLOR_TEXTO_OFF = (110, 110, 110)    # gris: capa desactivada
COLOR_TEXTO_PAUSA = (120, 180, 255)  # naranja suave: modo pausa

ANCHO_PANEL = 240        # ancho normal
ANCHO_PANEL_PAUSA = 300  # ancho cuando está pausado (más info)
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
    Dibuja sobre `frame` los landmarks de las manos detectadas (puede haber
    más de una mano si NUM_HANDS > 1), respetando el toggle `estado["mano"]`.

    `resultado` es el objeto que devuelve `landmarker.detect_for_video(...)`.
    """
    if not estado["mano"] or not resultado.hand_landmarks:
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


# =============================================================================
# PANEL DE INFORMACIÓN
# =============================================================================

def dibujar_panel_info(frame, fps, estado):
    """
    Dibuja (si `estado["panel"]` es True) un panel semitransparente en la
    esquina superior izquierda con:
        - los FPS suavizados
        - el estado ON/OFF del dibujo de la mano
        - el estado de pausa

    Si el panel está desactivado, esta función no dibuja nada y retorna
    inmediatamente.
    """
    if not estado["panel"]:
        return

    filas_toggle = [
        ("[h] Mano", estado["mano"]),
    ]

    # Geometría del panel
    pad = 12
    alto_fps = 22
    alto_linea = 20
    alto_panel = pad * 3 + alto_fps + 2 * alto_linea  # toggle + pausa

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

def _dibujar_bloque_landmarks(frame, x1, y, x2, y2, titulo, landmarks):
    """
    Helper interno: dibuja un bloque de landmarks en el panel de pausa.

    Devuelve la coordenada `y` siguiente (después del bloque), o None si
    ya no hay espacio vertical y hay que cortar el panel.
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

    for j, lm in enumerate(landmarks):
        if y + 12 > y2 - pad:
            cv2.putText(
                frame, "...", (x1 + pad, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
            )
            return None  # no hay más espacio

        texto = f"{j:3d}  {lm.x:.2f}  {lm.y:.2f}  {lm.z:.2f}"
        cv2.putText(
            frame, texto, (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 14

    return y + 6  # separación entre bloques


def dibujar_panel_pausa(frame, resultado, estado):
    """
    Cuando la captura está pausada, muestra en el lado derecho de la pantalla
    la información disponible del frame congelado, LIMITADA a las capas
    que están visibles según `estado`.

    Para cada mano detectada, muestra:
        - Lateralidad (Left / Right) + confianza de esa clasificación
        - Los 21 landmarks con sus coordenadas (x, y, z)

    Si la capa de mano está apagada (con `h`), este panel no dibuja nada.
    """
    if not estado["mano"] or not resultado.hand_landmarks:
        return

    alto_frame, ancho_frame, _ = frame.shape

    pad = 12
    ancho_detalle = 300
    x1 = ancho_frame - ancho_detalle - 10
    y1 = 10
    x2 = ancho_frame - 10
    y2 = alto_frame - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    y = y1 + pad
    cv2.putText(frame, "DETALLE (en pausa)", (x1 + pad, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO, 1, cv2.LINE_AA)
    y += 30

    for i, landmarks in enumerate(resultado.hand_landmarks):
        # Lateralidad + confianza
        lado = "?"
        confianza_txt = ""
        if resultado.handedness and i < len(resultado.handedness):
            categoria = resultado.handedness[i][0]
            lado = "Izquierda" if categoria.category_name == "Left" else "Derecha"
            confianza_txt = f" ({categoria.score:.2f})"

        titulo = f"MANO {i + 1} - {lado}{confianza_txt}"
        nuevo_y = _dibujar_bloque_landmarks(
            frame, x1, y, x2, y2,
            titulo,
            landmarks,
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
    elif tecla == ord("h"):
        estado["mano"] = not estado["mano"]
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
        print("https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker")
        print("y colocalo junto a este script.")
        return

    # Configuración del detector
    opciones = vision.HandLandmarkerOptions(
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

    # Estado inicial de los toggles
    estado_visualizacion = {
        "panel": MOSTRAR_PANEL_INICIAL,
        "mano": DIBUJAR_MANO_INICIAL,
        "pausa": False,
    }

    contador_fps = ContadorFPS(INTERVALO_ACTUALIZACION_FPS)

    print("Controles: [q]/[ESC] salir | [i] panel | [h] mano | [ESPACIO] pausa")

    try:
        try:
            landmarker = vision.HandLandmarker.create_from_options(opciones)
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

                # Detección de mano(s)
                resultado = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                # Dibujar sólo si el toggle de mano está activado
                dibujar_landmarks(frame, resultado, estado_visualizacion)

                # FPS suavizados + panel de información
                fps = contador_fps.actualizar()
                dibujar_panel_info(frame, fps, estado_visualizacion)

                # Si estamos en pausa, mostramos el panel detallado a la derecha
                # (respetando si la capa de mano está visible)
                if estado_visualizacion["pausa"]:
                    dibujar_panel_pausa(frame, resultado, estado_visualizacion)

                cv2.imshow(
                    "03 - Puntos de la Mano (q para salir)",
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