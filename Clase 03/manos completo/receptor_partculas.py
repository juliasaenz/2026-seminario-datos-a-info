"""
===============================================================================
 RECEPTOR OSC + PARTÍCULAS (Python + OpenCV)
===============================================================================

Este programa RECIBE los mensajes OSC que manda "03_puntos_mano_osc.py" y
los convierte en una visualización de partículas


CÓMO PROBARLO
-------------
  1) Ejecutar este programa (queda esperando datos).
  2) En otra terminal, ejecutar 03_puntos_mano_osc.py.
  3) Mover las manos frente a la cámara.

QUÉ HACE CADA DATO (se puede cambiar en la función `procesar_manos`)
--------------------------------------------------------------------
  palma        -> de dónde salen las partículas
  apertura     -> cuántas partículas salen y a qué velocidad
  inclinacion  -> color (girá la mano para recorrer el arco iris)
  cercania     -> tamaño de las partículas
  pinza        -> atrae las partículas de vuelta hacia la palma
  landmarks    -> se dibuja el esqueleto de la mano (tecla [l])

CONTROLES
---------
  q / ESC  -> salir
  l        -> mostrar / ocultar el esqueleto de la mano
  i        -> mostrar / ocultar el texto de información

CÓMO FUNCIONA LA RECEPCIÓN OSC
------------------------------------------------------------------
    1) Se abre un socket en un puerto ("bind") y se lo configura como
       NO BLOQUEANTE. Eso significa que si no hay nada para leer, el
       programa no se queda esperando: sigue de largo.
    2) En CADA cuadro del bucle principal se llama a `leer_mensajes_osc()`,
       que lee todos los paquetes que llegaron desde el cuadro anterior.
    3) Cada mensaje se decodifica con python-osc y se guarda en el
       diccionario `manos` (función `al_recibir`).
    4) Después se dibuja usando lo guardado en `manos`.

  Todo ocurre en orden, una cosa después de la otra: recibir -> procesar
  -> dibujar
===============================================================================
"""

import colorsys
import math
import random
import socket
import time

import cv2
import numpy as np
from pythonosc.osc_packet import OscPacket


# =============================================================================
# 1) PARÁMETROS EDITABLES
# =============================================================================

# --- Red ---------------------------------------------------------------------
# "127.0.0.1" = solo recibe desde esta misma computadora
IP_ESCUCHA = "127.0.0.1"
PUERTO_OSC = 9000          # el mismo que OSC_PUERTO en el emisor

# --- Ventana -----------------------------------------------------------------
ANCHO_VENTANA = 720
ALTO_VENTANA = 480

# --- Manos -------------------------------------------------------------------
# Si pasan más de estos segundos sin recibir datos de una mano, se considera que desapareció
TIMEOUT_MANO = 0.5

# --- Partículas --------------------------------------------------------------
MAX_PARTICULAS = 2500
EMISION_MAX = 10           # partículas por frame con la mano bien abierta
VELOCIDAD_MIN = 1.0        # velocidad inicial con la mano cerrada
VELOCIDAD_MAX = 7.0        # velocidad inicial con la mano abierta
RADIO_MIN = 2              # tamaño con la mano lejos
RADIO_MAX = 10             # tamaño con la mano cerca
VIDA_MIN = 40              # duración de cada partícula (en frames)
VIDA_MAX = 90
FUERZA_ATRACCION = 0.9     # qué tan fuerte atrae la pinza
ROZAMIENTO = 0.98          # 1.0 = sin frenado; menor = frena más rápido

# Estela: cada frame se oscurece la imagen anterior multiplicándola por este
# número. Cerca de 1.0 = estela larga; cerca de 0.5 = casi sin estela.
ESTELA = 0.86

# --- Colores (BGR) -----------------------------------------------------------
COLOR_FONDO = (0, 0, 0)
COLOR_ESQUELETO = (200, 200, 200)
COLOR_TEXTO = (230, 230, 230)

# Conexiones entre los 21 landmarks para dibujar el esqueleto de la mano
CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),            # índice
    (5, 9), (9, 10), (10, 11), (11, 12),       # medio
    (9, 13), (13, 14), (14, 15), (15, 16),     # anular
    (13, 17), (17, 18), (18, 19), (19, 20),    # meñique
    (0, 17),                                   # borde de la palma
]


# =============================================================================
# 2) RECEPCIÓN OSC
# =============================================================================

# Acá se guarda lo último que llegó de cada mano. Estructura:
#   manos = {
#       "derecha":   {"apertura": (0.8,), "palma": (0.5, 0.4), ..., "t": 1712.3},
#       "izquierda": {...},
#   }
# Cada valor se guarda como TUPLA, porque un mensaje OSC puede traer varios
# argumentos (por ejemplo "palma" trae x e y).
manos = {}


def abrir_socket_osc():
    """
    Abre el puerto UDP donde van a llegar los mensajes OSC.
    setblocking(False) = si no hay datos, no se queda esperando.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP_ESCUCHA, PUERTO_OSC))
    sock.setblocking(False)
    return sock


def al_recibir(direccion, argumentos):
    """
    Se llama una vez por cada mensaje OSC recibido.

    `direccion`  -> texto, por ejemplo "/mano/derecha/apertura"
    `argumentos` -> los valores del mensaje, por ejemplo [0.73]

    Separamos la dirección en partes:
        "/mano/derecha/apertura" -> ["mano", "derecha", "apertura"]
    """
    partes = direccion.strip("/").split("/")

    # Solo nos interesan los mensajes con forma /mano/<lado>/<dato>
    if len(partes) < 3 or partes[0] != "mano":
        return

    lado, dato = partes[1], partes[2]

    # Si es la primera vez que vemos esta mano, creamos su espacio
    mano = manos.setdefault(lado, {})
    mano[dato] = tuple(argumentos)
    mano["t"] = time.time()     # cuándo llegó el último mensaje


def leer_mensajes_osc(sock):
    """
    Lee TODOS los paquetes que están esperando en el socket y los manda a
    `al_recibir`. Se llama una vez por cuadro.
    """
    while True:
        try:
            datos, _origen = sock.recvfrom(65535)
        except BlockingIOError:
            break               # no hay más paquetes por ahora

        try:
            paquete = OscPacket(datos)      # decodifica los bytes
        except Exception:
            continue                        # paquete que no es OSC: lo ignoramos

        # Un paquete puede traer uno o varios mensajes
        for mensaje_con_tiempo in paquete.messages:
            mensaje = mensaje_con_tiempo.message
            al_recibir(mensaje.address, mensaje.params)


def leer(datos, nombre, defecto=0.0):
    """Devuelve el primer valor del dato `nombre`, o `defecto` si no llegó."""
    valor = datos.get(nombre)
    if valor is None or len(valor) == 0:
        return defecto
    return valor[0]


# =============================================================================
# 3) PARTÍCULAS
# =============================================================================

class Particula:
    """
    Una partícula: nace en un punto, sale en una dirección al azar,
    se va frenando y se apaga con el tiempo.
    """

    def __init__(self, x, y, velocidad, color, radio, dueno):
        angulo = random.uniform(0, 2 * math.pi)
        rapidez = velocidad * random.uniform(0.3, 1.0)

        self.x = x
        self.y = y
        self.vx = math.cos(angulo) * rapidez
        self.vy = math.sin(angulo) * rapidez
        self.color = color
        self.radio = radio
        self.dueno = dueno                       # "derecha" o "izquierda"
        self.vida_max = random.randint(VIDA_MIN, VIDA_MAX)
        self.vida = self.vida_max

    def actualizar(self, atractor=None):
        """
        Mueve la partícula un paso.
        `atractor` es None o una tupla (x, y, fuerza) hacia donde se la empuja.
        """
        if atractor is not None:
            ax, ay, fuerza = atractor
            dx, dy = ax - self.x, ay - self.y
            distancia = max(math.hypot(dx, dy), 1.0)
            self.vx += dx / distancia * fuerza
            self.vy += dy / distancia * fuerza

        self.x += self.vx
        self.y += self.vy
        self.vx *= ROZAMIENTO
        self.vy *= ROZAMIENTO
        self.vida -= 1

    def viva(self):
        return self.vida > 0

    def dibujar(self, lienzo):
        # A medida que pierde vida, se vuelve más oscura y más chica
        factor = self.vida / self.vida_max
        color = tuple(int(c * factor) for c in self.color)
        radio = max(1, int(self.radio * factor))
        cv2.circle(lienzo, (int(self.x), int(self.y)), radio, color, -1, cv2.LINE_AA)


# =============================================================================
# 4) UTILIDADES
# =============================================================================

def mapear(valor, entrada_min, entrada_max, salida_min, salida_max):
    """Convierte `valor` del rango de entrada al rango de salida (recortado)."""
    t = (valor - entrada_min) / (entrada_max - entrada_min)
    t = max(0.0, min(1.0, t))
    return salida_min + t * (salida_max - salida_min)


def color_desde_tono(tono):
    """Tono 0-1 (arco iris) -> color BGR de OpenCV."""
    r, g, b = colorsys.hsv_to_rgb(tono % 1.0, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


# =============================================================================
# 5) LÓGICA: DE LOS DATOS OSC A LAS PARTÍCULAS
# =============================================================================
# ESTA ES LA FUNCIÓN PARA EXPERIMENTAR: acá se decide qué dato controla qué

def procesar_manos(particulas):
    """
    Recorre las manos activas, crea partículas nuevas según sus datos y
    devuelve un diccionario de "atractores" (uno por mano con pinza)
    """
    ahora = time.time()
    atractores = {}

    for lado, datos in manos.items():

        # Ignorar manos que dejaron de enviar datos
        if ahora - datos.get("t", 0) > TIMEOUT_MANO:
            continue

        # Sin posición de palma no podemos dibujar nada
        palma = datos.get("palma")
        if palma is None or len(palma) < 2:
            continue

        # --- Leer los datos (valores de 0 a 1, salvo inclinación) -------------
        x = palma[0] * ANCHO_VENTANA           # de 0-1 a píxeles
        y = palma[1] * ALTO_VENTANA
        apertura = leer(datos, "apertura")
        pinza = leer(datos, "pinza")
        cercania = leer(datos, "cercania", 0.5)
        inclinacion = leer(datos, "inclinacion")   # -180 a 180 grados

        # --- Traducir datos a propiedades visuales ----------------------------
        # apertura -> cantidad y velocidad
        cantidad = int(1 + apertura * EMISION_MAX)
        velocidad = mapear(apertura, 0, 1, VELOCIDAD_MIN, VELOCIDAD_MAX)

        # cercania -> tamaño
        radio = mapear(cercania, 0, 1, RADIO_MIN, RADIO_MAX)

        # inclinacion -> color (de -90 a 90 grados recorre 80% del arco iris);
        # la mano derecha se desplaza medio arco iris para diferenciarse
        tono = mapear(inclinacion, -90, 90, 0.0, 0.8)
        if lado == "derecha":
            tono += 0.5
        color = color_desde_tono(tono)

        # pinza -> fuerza con la que se atraen las partículas de esta mano
        if pinza > 0.3:
            atractores[lado] = (x, y, pinza * FUERZA_ATRACCION)

        # --- Crear las partículas nuevas --------------------------------------
        for _ in range(cantidad):
            if len(particulas) >= MAX_PARTICULAS:
                break
            particulas.append(Particula(x, y, velocidad, color, radio, lado))

    return atractores


# =============================================================================
# 6) DIBUJO DEL ESQUELETO Y DEL TEXTO
# =============================================================================

def dibujar_esqueletos(lienzo):
    """Dibuja los 21 puntos de cada mano activa usando el dato 'landmarks'"""
    ahora = time.time()

    for lado, datos in manos.items():
        if ahora - datos.get("t", 0) > TIMEOUT_MANO:
            continue

        valores = datos.get("landmarks")
        if valores is None or len(valores) != 63:
            continue

        # Los 63 valores vienen como x0 y0 z0 x1 y1 z1 ...
        # Los agrupamos de a tres y nos quedamos con x e y (en píxeles)
        puntos = [
            (int(valores[i] * ANCHO_VENTANA), int(valores[i + 1] * ALTO_VENTANA))
            for i in range(0, 63, 3)
        ]

        for a, b in CONEXIONES_MANO:
            cv2.line(lienzo, puntos[a], puntos[b], COLOR_ESQUELETO, 1, cv2.LINE_AA)
        for p in puntos:
            cv2.circle(lienzo, p, 3, COLOR_ESQUELETO, -1, cv2.LINE_AA)


def dibujar_info(lienzo):
    """Escribe en pantalla los valores que se están recibiendo."""
    ahora = time.time()
    lineas = [f"Escuchando OSC en {IP_ESCUCHA}:{PUERTO_OSC}"]

    activas = 0
    for lado, datos in manos.items():
        if ahora - datos.get("t", 0) > TIMEOUT_MANO:
            continue
        activas += 1
        dedos = int(leer(datos, "cantidad_dedos"))
        lineas.append(
            f"{lado:10s} apertura {leer(datos, 'apertura'):.2f}  "
            f"pinza {leer(datos, 'pinza'):.2f}  "
            f"incl {leer(datos, 'inclinacion'):6.1f}  "
            f"cerca {leer(datos, 'cercania'):.2f}  dedos {dedos}"
        )

    if activas == 0:
        lineas.append("Esperando manos... (ejecutar 03_puntos_mano_osc.py)")

    for k, texto in enumerate(lineas):
        cv2.putText(lienzo, texto, (15, 25 + k * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO, 1, cv2.LINE_AA)


# =============================================================================
# 7) MAIN
# =============================================================================

def main():
    sock = abrir_socket_osc()
    print(f"Escuchando OSC en {IP_ESCUCHA}:{PUERTO_OSC}")
    print("Controles: [q]/[ESC] salir | [l] esqueleto | [i] info")

    lienzo = np.zeros((ALTO_VENTANA, ANCHO_VENTANA, 3), dtype=np.uint8)
    lienzo[:] = COLOR_FONDO

    particulas = []
    mostrar_esqueleto = True
    mostrar_info = True

    try:
        while True:
            # 1) Leer los mensajes OSC que llegaron desde el cuadro anterior
            leer_mensajes_osc(sock)

            # 2) Estela: oscurecemos lo dibujado en el frame anterior
            lienzo = cv2.convertScaleAbs(lienzo, alpha=ESTELA)

            # 3) Crear partículas nuevas según los datos recibidos
            atractores = procesar_manos(particulas)

            # 4) Mover y dibujar todas las partículas
            for p in particulas:
                p.actualizar(atractores.get(p.dueno))
                p.dibujar(lienzo)

            # 5) Eliminar las que se apagaron
            particulas = [p for p in particulas if p.viva()]

            # 6) Capas opcionales encima
            if mostrar_esqueleto:
                dibujar_esqueletos(lienzo)
            if mostrar_info:
                dibujar_info(lienzo)

            cv2.imshow("Receptor OSC - Particulas (q para salir)", lienzo)

            # 7) Teclado
            tecla = cv2.waitKey(16) & 0xFF      # ~60 cuadros por segundo
            if tecla == ord("q") or tecla == 27:
                break
            elif tecla == ord("l"):
                mostrar_esqueleto = not mostrar_esqueleto
            elif tecla == ord("i"):
                mostrar_info = not mostrar_info

    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()