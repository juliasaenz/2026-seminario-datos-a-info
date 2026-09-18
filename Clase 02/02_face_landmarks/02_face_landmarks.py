"""
===============================================================================
 02 - PUNTOS FACIALES (Face Landmarks)
===============================================================================

Este script usa la API moderna "MediaPipe Tasks" para detectar puntos
faciales en tiempo real.

Face Landmarker devuelve:
  - Una malla completa de 478 puntos en 3D sobre el rostro.
  - 52 "blendshapes": coeficientes que describen la expresión facial
    (sonrisa, ceño fruncido, boca abierta, etc.).
  - Opcionalmente: una matriz de transformación facial para efectos 3D.

SOBRE LOS BLENDSHAPES
---------------------
Los blendshapes son 52 coeficientes (0.0 a 1.0) que representan
expresiones faciales. Algunos ejemplos:
  - "mouthSmileLeft" / "mouthSmileRight"  (sonrisa)
  - "eyeBlinkLeft" / "eyeBlinkRight"      (parpadeo)
  - "jawOpen"                             (boca abierta)
  - "browDownLeft" / "browDownRight"      (ceño fruncido)

Algunos blendshapes vienen en pares Izq/Der (independientes), otros son
únicos (como jawOpen). En total hay 52, y el primero es "_neutral"
(con guion bajo), que indica cuánto se parece el rostro a la expresión
neutra.

DESCARGA DEL MODELO
-------------------
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

Buscar la sección "Models" y descargar:
    face_landmarker.task
y colocarlo en la misma carpeta que este script.

CONTROLES (con la ventana de la cámara activa)
-----------------------------------------------
    q / ESC   -> cerrar el programa
    i         -> mostrar / ocultar el panel de información
    b         -> mostrar / ocultar el panel de BLENDSHAPES en vivo
    m         -> mostrar / ocultar la MALLA facial (triangulación)
    c         -> mostrar / ocultar los CONTORNOS (ojos, cejas, labios, cara)
    r         -> mostrar / ocultar el IRIS
    ESPACIO   -> pausar / reanudar la captura
                 (al pausar, se congela el último frame y se analiza
                 ese frame con detalle: los 478 landmarks con sus
                 coordenadas x, y, z, y los 52 blendshapes con su valor)

NOTA: al pausar, el panel de blendshapes en vivo se apaga automáticamente.
En pausa se muestra en su lugar el panel de detalle, que ya incluye los
blendshapes. Al reanudar, el toggle queda como estaba: apagado.
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

MODEL_FILENAME = "face_landmarker.task"
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME

# Confianza mínima (entre 0.0 y 1.0) para que el modelo dé por detectado un rostro
MIN_FACE_DETECTION_CONFIDENCE = 0.5

# Confianza mínima para considerar que el rostro sigue "presente"
# una vez detectado (se usa frame a frame, no sólo en la detección inicial)
MIN_FACE_PRESENCE_CONFIDENCE = 0.5

# Confianza mínima para el seguimiento (tracking) de los puntos entre frames
#   Valores más bajos = tracking más "suelto" (puede perderse más fácil);
#   valores más altos = más estricto (puede tardar más en reengancharse)
MIN_TRACKING_CONFIDENCE = 0.5

# Cantidad máxima de rostros a detectar simultáneamente
# Con más de 1, el rendimiento baja notoriamente
NUM_FACES = 1

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
MOSTRAR_BLENDSHAPES_INICIAL = False
DIBUJAR_MALLA_INICIAL = True
DIBUJAR_CONTORNOS_INICIAL = True
DIBUJAR_IRIS_INICIAL = True

# Cada cuántos segundos se recalcula el número de FPS
INTERVALO_ACTUALIZACION_FPS = 0.5

# Colores en formato BGR (el que usa OpenCV, no RGB).
COLOR_PANEL = (30, 30, 30)
COLOR_TEXTO = (240, 240, 240)
COLOR_TEXTO_TENUE = (170, 170, 170)
COLOR_TEXTO_ON = (120, 220, 120)     # verde suave: capa activada
COLOR_TEXTO_OFF = (110, 110, 110)    # gris: capa desactivada
COLOR_TEXTO_PAUSA = (120, 180, 255)  # naranja suave: modo pausa

# Colores de los elementos faciales
COLOR_MALLA = (80, 80, 80)     # gris para la triangulación
COLOR_CONTORNOS = (0, 255, 0)  # verde para contornos
COLOR_IRIS = (255, 0, 0)       # azul para el iris

ANCHO_PANEL = 240        # ancho del panel de info
ANCHO_PANEL_PAUSA = 320  # ancho cuando está pausado (más info)
ALPHA_PANEL = 0.65       # transparencia del fondo (0 = invisible, 1 = opaco)

# Separación horizontal entre el panel de info y el panel de la derecha
SEPARACION_PANELES = 10

# Ancho del panel de la derecha (blendshapes en vivo o detalle en pausa).
# Comparten el mismo espacio físico, así que comparten el ancho.
ANCHO_PANEL_DERECHA = 300

# Cuántos landmarks del rostro mostrar en el panel de pausa.
# 478 no entran en pantalla, así que recortamos.
MAX_LANDMARKS_ROSTRO_PAUSA = 50


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
    Dibuja sobre `frame` la malla facial y los contornos, respetando los
    toggles `estado["malla"]`, `estado["contornos"]` y `estado["iris"]`.

    `resultado` es el objeto que devuelve `landmarker.detect_for_video(...)`.
    """
    if not resultado.face_landmarks:
        return

    mp_drawing = mp.tasks.vision.drawing_utils

    for landmarks in resultado.face_landmarks:
        # --------------------------------------------------------------------
        # Malla completa (triangulación)
        # --------------------------------------------------------------------
        if estado["malla"]:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=COLOR_MALLA, thickness=1
                ),
            )

        # --------------------------------------------------------------------
        # Contornos (ojos, cejas, labios, óvalo facial)
        # --------------------------------------------------------------------
        if estado["contornos"]:
            grupos = [
                mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
                mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
                mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
                mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
                mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
                mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
            ]
            for connections in grupos:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=COLOR_CONTORNOS, thickness=2
                    ),
                )

        # ---------------------------------------------------------------------
        # Iris
        # ---------------------------------------------------------------------
        if estado["iris"]:
            for connections in (
                mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            ):
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=COLOR_IRIS, thickness=2
                    ),
                )


# =============================================================================
# PANEL DE INFORMACIÓN BÁSICO (izquierda)
# =============================================================================

def dibujar_panel_info(frame, fps, estado):
    """
    Dibuja (si `estado["panel"]` es True) un panel semitransparente en la
    esquina superior izquierda con:
        - los FPS suavizados
        - el estado ON/OFF de cada capa (malla, contornos, iris, blendshapes)
        - el estado de pausa

    Es un panel básico: NO muestra blendshapes. Para verlos en vivo, se
    activa el panel de blendshapes con la tecla `b`.
    """
    if not estado["panel"]:
        return

    filas_toggle = [
        ("[m] Malla", estado["malla"]),
        ("[c] Contornos", estado["contornos"]),
        ("[r] Iris", estado["iris"]),
        ("[b] Blendshapes", estado["blendshapes"]),
    ]

    # Geometría del panel
    pad = 12
    alto_fps = 22
    alto_linea = 20
    n_filas = len(filas_toggle) + 1  # toggles + pausa
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
# PANEL DE LA DERECHA
# =============================================================================
# Este panel es compartido por dos vistas:
#   - En vivo, si `estado["blendshapes"]` está activo: muestra los 52
#     blendshapes con su valor actual.
#   - En pausa: muestra los landmarks + los 52 blendshapes.
#
# Las dos vistas comparten posición y tamaño, pero nunca se dibujan juntas:
# al pausar, el toggle de blendshapes se apaga automáticamente.

def _dibujar_bloque_landmarks(frame, x1, y, x2, y2, titulo, landmarks, max_mostrados=None):
    """
    Helper interno: dibuja un bloque de landmarks (x, y, z). Recibe una
    LISTA PLANA de landmarks (los 478 puntos de un rostro), no una lista
    de listas.

    Devuelve la coordenada `y` siguiente, o None si ya no hay espacio.
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
            return None
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

    return y + 6


def _dibujar_bloque_blendshapes(frame, x1, y, x2, y2, titulo, blendshapes):
    """
    Helper interno: dibuja la lista completa de blendshapes (nombre + valor)
    tal como los devuelve el modelo, sin traducción.

    `blendshapes` es la lista de `Category` que viene en
    `resultado.face_blendshapes[0]`.

    Devuelve la coordenada `y` siguiente, o None si ya no hay espacio.
    """
    pad = 12
    cv2.putText(
        frame, titulo, (x1 + pad, y + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXTO, 1, cv2.LINE_AA,
    )
    y += 18

    for bs in blendshapes:
        if y + 12 > y2 - pad:
            cv2.putText(
                frame, "...", (x1 + pad, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
            )
            return None

        texto = f"{bs.category_name}: {bs.score:.2f}"
        cv2.putText(
            frame, texto, (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXTO_TENUE, 1, cv2.LINE_AA,
        )
        y += 14

    return y + 6


def _geometria_panel_derecha(frame, estado):
    """
    Devuelve (x1, y1, x2, y2) del panel de la derecha.
    Comparte posición con el panel de pausa.
    """
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


def dibujar_panel_blendshapes_vivo(frame, resultado, estado):
    """
    En vivo (y si `estado["blendshapes"]` está activo), muestra el panel
    de la derecha con los 52 blendshapes tal como los devuelve el modelo.
    """
    if estado["pausa"] or not estado["blendshapes"]:
        return

    x1, y1, x2, y2 = _geometria_panel_derecha(frame, estado)
    _dibujar_fondo_panel_derecha(frame, x1, y1, x2, y2)

    pad = 12
    y = y1 + pad
    cv2.putText(
        frame, "BLENDSHAPES (en vivo)", (x1 + pad, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO, 1, cv2.LINE_AA,
    )
    y += 30

    if not resultado.face_blendshapes:
        cv2.putText(
            frame, "Sin rostro", (x1 + pad, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXTO_OFF, 1, cv2.LINE_AA,
        )
        return

    _dibujar_bloque_blendshapes(
        frame, x1, y, x2, y2,
        "",
        resultado.face_blendshapes[0],
    )


def dibujar_panel_pausa(frame, resultado, estado):
    """
    En pausa, muestra el panel de la derecha con:
      - Los landmarks del rostro (si alguna capa de dibujo está activa),
        recortados a MAX_LANDMARKS_ROSTRO_PAUSA.
      - Los 52 blendshapes tal como los devuelve el modelo.

    Este panel comparte posición con el de blendshapes en vivo, pero nunca
    se dibujan juntos: al pausar, el toggle de blendshapes se apaga.
    """
    if not estado["pausa"]:
        return

    if not resultado.face_landmarks:
        return

    x1, y1, x2, y2 = _geometria_panel_derecha(frame, estado)
    _dibujar_fondo_panel_derecha(frame, x1, y1, x2, y2)

    pad = 12
    y = y1 + pad
    cv2.putText(
        frame, "DETALLE (en pausa)", (x1 + pad, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO, 1, cv2.LINE_AA,
    )
    y += 30

    # --- Landmarks (sólo si alguna capa de dibujo está activa) ---
    if estado["malla"] or estado["contornos"] or estado["iris"]:
        for i, landmarks in enumerate(resultado.face_landmarks):
            titulo = f"ROSTRO {i + 1} ({len(landmarks)} landmarks)"
            nuevo_y = _dibujar_bloque_landmarks(
                frame, x1, y, x2, y2,
                titulo,
                landmarks,
                max_mostrados=MAX_LANDMARKS_ROSTRO_PAUSA,
            )
            if nuevo_y is None:
                return
            y = nuevo_y

    # --- Blendshapes ---
    if resultado.face_blendshapes:
        _dibujar_bloque_blendshapes(
            frame, x1, y, x2, y2,
            f"BLENDSHAPES ({len(resultado.face_blendshapes[0])})",
            resultado.face_blendshapes[0],
        )


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
    elif tecla == ord("b"):
        estado["blendshapes"] = not estado["blendshapes"]
    elif tecla == ord("m"):
        estado["malla"] = not estado["malla"]
    elif tecla == ord("c"):
        estado["contornos"] = not estado["contornos"]
    elif tecla == ord("r"):
        estado["iris"] = not estado["iris"]
    elif tecla == ord(" "):
        # Al pausar, apagamos el panel de blendshapes en vivo.
        # Al reanudar, queda apagado (no lo volvemos a encender solo).
        if not estado["pausa"]:
            estado["blendshapes"] = False
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
        print("https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker")
        print("y colocalo junto a este script.")
        return

    # Configuración del detector
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
        "blendshapes": MOSTRAR_BLENDSHAPES_INICIAL,
        "malla": DIBUJAR_MALLA_INICIAL,
        "contornos": DIBUJAR_CONTORNOS_INICIAL,
        "iris": DIBUJAR_IRIS_INICIAL,
        "pausa": False,
    }

    contador_fps = ContadorFPS(INTERVALO_ACTUALIZACION_FPS)

    print("Controles: [q]/[ESC] salir | [i] panel | [b] blendshapes | [m] malla | [c] contornos | [r] iris | [ESPACIO] pausa")

    try:
        try:
            landmarker = vision.FaceLandmarker.create_from_options(opciones)
        except Exception as e:
            print(f"Error al crear el detector: {e}")
            print("Verificá que el archivo del modelo no esté corrupto.")
            return

        with landmarker:
            # Frame congelado cuando está en pausa.
            frame_congelado = None

            while cap.isOpened():
                # ------------------------------------------------------------------
                # ¿De dónde sacamos el frame?
                # ------------------------------------------------------------------
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

                # Timestamp real
                timestamp_ms = int(time.time() * 1000)

                # Detectar
                resultado = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                # Dibujar
                dibujar_landmarks(frame, resultado, estado_visualizacion)

                # FPS
                fps = contador_fps.actualizar()

                # Panel de info (izquierda)
                dibujar_panel_info(frame, fps, estado_visualizacion)

                # Panel de la derecha:
                #   - en vivo: blendshapes (si el toggle está activo)
                #   - en pausa: detalle (landmarks + blendshapes)
                # Nunca se dibujan los dos: la pausa apaga el toggle.
                dibujar_panel_blendshapes_vivo(frame, resultado, estado_visualizacion)
                dibujar_panel_pausa(frame, resultado, estado_visualizacion)

                cv2.imshow(
                    "02 - Puntos Faciales (q para salir)",
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