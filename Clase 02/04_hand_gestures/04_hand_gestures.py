"""
===============================================================================
 04 - RECONOCIMIENTO DE GESTOS (Hand Gesture Recognition)
===============================================================================

Este script usa la API moderna "MediaPipe Tasks" para reconocer gestos de
la mano en tiempo real.

El modelo detecta la mano, encuentra sus 21 puntos clave, y además
clasifica QUÉ gesto está haciendo. Los gestos predefinidos son:
  - None:         Sin gesto reconocido
  - Closed_Fist:  Puño cerrado
  - Open_Palm:    Palma abierta
  - Pointing_Up:  Dedo índice apuntando hacia arriba
  - Thumb_Down:   Pulgar hacia abajo
  - Thumb_Up:     Pulgar hacia arriba
  - Victory:      Seña de victoria (dos dedos)
  - ILoveYou:     Seña "te amo" (lengua de señas)

DESCARGA DEL MODELO
-------------------
https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer

Buscar la sección "Models" y descargar:
    gesture_recognizer.task

CONTROLES (con la ventana de la cámara activa)
-----------------------------------------------
    q / ESC   -> cerrar el programa
    i         -> mostrar / ocultar el panel de información
    h         -> mostrar / ocultar los landmarks de la mano
    ESPACIO   -> pausar / reanudar la captura
                 (al pausar, se congela el último frame y se analiza
                 ese frame con detalle: landmarks, gestos y coordenadas)
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

MODEL_FILENAME = "gesture_recognizer.task"
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME

# Confianza mínima (entre 0.0 y 1.0) para que el modelo dé por detectada una mano
MIN_HAND_DETECTION_CONFIDENCE = 0.5

# Confianza mínima para considerar que la mano sigue "presente"
# una vez detectada (se usa frame a frame, no sólo en la detección inicial)
MIN_HAND_PRESENCE_CONFIDENCE = 0.5

# Confianza mínima para el seguimiento (tracking) de los puntos entre frames
#   Valores más bajos = tracking más "suelto" (puede perderse más fácil);
#   valores más altos = más estricto (puede tardar más en reengancharse)
MIN_TRACKING_CONFIDENCE = 0.5

# Cantidad máxima de manos a detectar simultáneamente
NUM_HANDS = 4

# Índice de la cámara a usar. Si hay más de una cámara conectada, probar 0, 1, 2... hasta encontrar la correcta
CAMARA_INDEX = 0

# Resolución de captura pedida a la cámara (la cámara puede no respetarla exactamente; depende del driver / hardware)
ANCHO_CAPTURA = 640
ALTO_CAPTURA = 480


# =============================================================================
# 2) PARÁMETROS EDITABLES DEL PANEL Y LA VISUALIZACIÓN
# =============================================================================
# Todo lo relacionado a QUÉ se dibuja y CÓMO se ve el panel de información

# Estado inicial al arrancar el programa
MOSTRAR_PANEL_INICIAL = True
DIBUJAR_MANO_INICIAL = True

# Cada cuántos segundos se recalcula el número de FPS
INTERVALO_ACTUALIZACION_FPS = 0.5

# Colores en formato BGR (el que usa OpenCV, no RGB).
COLOR_PANEL = (30, 30, 30)
COLOR_TEXTO = (240, 240, 240)
COLOR_TEXTO_TENUE = (170, 170, 170)
COLOR_TEXTO_ON = (120, 220, 120)   # verde suave: capa activada
COLOR_TEXTO_OFF = (110, 110, 110)  # gris: capa desactivada
COLOR_TEXTO_PAUSA = (120, 180, 255)  # naranja suave: modo pausa

ANCHO_PANEL = 240        # ancho normal
ANCHO_PANEL_PAUSA = 320  # ancho cuando está pausado (más info)
ALPHA_PANEL = 0.65       # transparencia del fondo (0 = invisible, 1 = opaco)


# Traducción de los nombres de gestos al español (para mostrar en el panel)
TRADUCCION_GESTOS = {
    "None":        "Sin gesto",
    "Closed_Fist": "Puño",
    "Open_Palm":   "Palma abierta",
    "Pointing_Up": "Apuntando",
    "Thumb_Down":  "Pulgar abajo",
    "Thumb_Up":    "Pulgar arriba",
    "Victory":     "Victoria",
    "ILoveYou":    "Te amo",
}


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

    `resultado` es el objeto que devuelve `recognizer.recognize_for_video(...)`.
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

def dibujar_panel_info(frame, fps, resultado, estado):
    """
    Dibuja (si `estado["panel"]` es True) un panel semitransparente en la
    esquina superior izquierda con:
        - los FPS suavizados
        - el estado ON/OFF del dibujo de la mano
        - el estado de pausa (si está pausado, se indica claramente)
        - la cantidad de manos detectadas
        - por cada mano: lateralidad + gesto reconocido + confianza

    Si el panel está desactivado, esta función no dibuja nada y retorna
    inmediatamente.
    """
    if not estado["panel"]:
        return

    # ----------------------------------------------------------------------
    # Armar las líneas de información
    # ----------------------------------------------------------------------
    lineas = []

    cantidad_manos = len(resultado.hand_landmarks) if resultado.hand_landmarks else 0
    lineas.append(f"Manos detectadas: {cantidad_manos}")

    if resultado.gestures:
        for i, gestos in enumerate(resultado.gestures):
            if not gestos:
                continue
            gesto = gestos[0]
            etiqueta = TRADUCCION_GESTOS.get(gesto.category_name, gesto.category_name)
            lineas.append(f"Mano {i + 1}: {etiqueta} ({gesto.score:.2f})")

    if cantidad_manos == 0:
        lineas.append("Sin manos")

    # ----------------------------------------------------------------------
    # Geometría del panel
    # ----------------------------------------------------------------------
    pad = 12
    alto_fps = 22
    alto_linea = 20
    alto_panel = pad * 3 + alto_fps + len(lineas) * alto_linea + 30  # +20 estado

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
    # Estado del toggle de mano (ON en verde, OFF en gris)
    # ----------------------------------------------------------------------
    color_estado = COLOR_TEXTO_ON if estado["mano"] else COLOR_TEXTO_OFF
    texto_estado = f"[h] Mano: {'ON' if estado['mano'] else 'OFF'}"
    cv2.putText(
        frame, texto_estado, (x1 + pad, y + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_estado, 1, cv2.LINE_AA,
    )
    y += alto_linea

    # ----------------------------------------------------------------------
    # Filas de información
    # ----------------------------------------------------------------------
    for linea in lineas:
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
        y += alto_linea

    # ----------------------------------------------------------------------
    # Estado de pausa (se muestra siempre, al final del panel)
    # ----------------------------------------------------------------------
    if estado["pausa"]:
        cv2.putText(
            frame,
            "[ESPACIO] EN PAUSA",
            (x1 + pad, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_TEXTO_PAUSA,
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "[ESPACIO] en vivo",
            (x1 + pad, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_TEXTO_OFF,
            1,
            cv2.LINE_AA,
        )


# =============================================================================
# PANEL DETALLADO EN PAUSA
# =============================================================================

def dibujar_panel_pausa(frame, resultado):
    """
    Cuando la captura está pausada, muestra en el lado derecho de la pantalla
    toda la información disponible de cada mano detectada:

        - Lateralidad (Left / Right)
        - Gesto reconocido + confianza
        - Los 21 landmarks con sus coordenadas (x, y, z)

    Es mucha información, por eso se muestra sólo en pausa y en un panel
    aparte, para no saturar la vista durante la captura en vivo.
    """
    if not resultado.hand_landmarks:
        return

    alto_frame, ancho_frame, _ = frame.shape

    # Panel a la derecha
    pad = 12
    ancho_detalle = 260
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

    # Por cada mano
    for i, landmarks in enumerate(resultado.hand_landmarks):
        # Encabezado de la mano
        lado = "?"
        if resultado.handedness and i < len(resultado.handedness):
            categoria = resultado.handedness[i][0].category_name
            lado = "Izquierda" if categoria == "Left" else "Derecha"

        gesto_txt = "?"
        if resultado.gestures and i < len(resultado.gestures) and resultado.gestures[i]:
            g = resultado.gestures[i][0]
            gesto_txt = f"{TRADUCCION_GESTOS.get(g.category_name, g.category_name)} ({g.score:.2f})"

        cv2.putText(
            frame, f"Mano {i + 1}: {lado}", (x1 + pad, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXTO, 1, cv2.LINE_AA,
        )
        y += 18
        cv2.putText(
            frame, f"Gesto: {gesto_txt}", (x1 + pad, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 20

        # Los 21 landmarks
        cv2.putText(
            frame, "Landmarks (x, y, z):", (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 16

        for j, lm in enumerate(landmarks):
            # Si se acaba el espacio vertical, cortamos
            if y + 12 > y2 - pad:
                cv2.putText(
                    frame, "...", (x1 + pad, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
                )
                return

            texto = f"{j:2d}: {lm.x:.2f}, {lm.y:.2f}, {lm.z:.2f}"
            cv2.putText(
                frame, texto, (x1 + pad, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
            )
            y += 14

        # Separación entre manos
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
        print("https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer")
        print("y colocalo junto a este script.")
        return

    # Configuración del detector
    opciones = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=python.BaseOptions.Delegate.CPU,
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
            recognizer = vision.GestureRecognizer.create_from_options(opciones)
        except Exception as e:
            print(f"Error al crear el detector: {e}")
            print("Verificá que el archivo del modelo no esté corrupto.")
            return

        with recognizer:
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

                # Detección + reconocimiento de gestos
                resultado = recognizer.recognize_for_video(
                    mp_image,
                    timestamp_ms,
                )

                # Dibujar sólo si el toggle de mano está activado
                dibujar_landmarks(frame, resultado, estado_visualizacion)

                # FPS suavizados + panel de información
                fps = contador_fps.actualizar()
                dibujar_panel_info(frame, fps, resultado, estado_visualizacion)

                # Si estamos en pausa, mostramos el panel detallado a la derecha
                if estado_visualizacion["pausa"]:
                    dibujar_panel_pausa(frame, resultado)

                cv2.imshow(
                    "04 - Gestos de la Mano (q para salir)",
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