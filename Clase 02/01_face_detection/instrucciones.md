# Guía: Detección de Rostros con MediaPipe Tasks

Esta guía acompaña al script `01_face_detection.py`. Explica qué es
MediaPipe, cómo funciona la detección de rostros por dentro, cómo ejecutar
el código y qué experimentos se pueden hacer en clase.

---

## Índice

- [Guía: Detección de Rostros con MediaPipe Tasks](#guía-detección-de-rostros-con-mediapipe-tasks)
  - [Índice](#índice)
  - [1. Detección de rostros: la idea general](#1-detección-de-rostros-la-idea-general)
  - [2. La API Tasks vs. la API clásica](#2-la-api-tasks-vs-la-api-clásica)
    - [API clásica: `mp.solutions`](#api-clásica-mpsolutions)
    - [API moderna: `mediapipe.tasks`](#api-moderna-mediapipetasks)
  - [3. ¿Qué devuelve el detector?](#3-qué-devuelve-el-detector)
  - [4. Cómo funciona el modelo por dentro](#4-cómo-funciona-el-modelo-por-dentro)
  - [5. Requisitos e instalación](#5-requisitos-e-instalación)
  - [6. Ejecutar el código](#6-ejecutar-el-código)

---

## 1. Detección de rostros: la idea general

**Detectar** un rostro ≠ **reconocer** un rostro.

Este script responde solamente: *"hay una cara en esta región de la imagen,
con estos puntos característicos y esta confianza"*. No sabe quién es.

Eso es importante: El modelo ve *formas*, no *personas*.

---

## 2. La API Tasks vs. la API clásica

MediaPipe tuvo (y todavía tiene) dos APIs:

### API clásica: `mp.solutions`

- Más simple
- Los modelos están incluidos en el paquete de `mediapipe`
- Está en modo "mantenimiento", Google ya no lo actualiza

### API moderna: `mediapipe.tasks`
- Requiere descargar el modelo aparte (un `.tflite`)
- Es la API recomendada hoy
- Permite cambiar el modelo según lo necesario

## 3. ¿Qué devuelve el detector?
Por cada rostro detectado en la imagen, `resultado.detections` contiene un objeto `Detection` con:
 - **Bounding Box**
   - `origin_x` y `origin_y`: esquina superior en píxeles
   - `width` y `height`: tamaño en píxeles
 - **Keypoints** (Puntos clave):
    Las coordenadas están normalozazdas entre 0.0 y 1.0
    - 0: ojo derecho
    - 1: ojo izquierdo
    - 2: punta de nariz
    - 3: centro de boca
    - 4: oreja derecha
    - 5: oreja izquierda
    > Este orden es convención del modelo, no de la biblioteca, por lo que puede cambiar según el modelo
- **Score** (Confianza):
    Un número entre 0.0 y 1.0 que indica cuán seguro está el modelo de que eso es una cara

## 4. Cómo funciona el modelo por dentro
`blaze_face_short_range.tflite` es una red neuronal llamada BlazeFace.

La idea general es:
1. La imagen se redimensiona a una resolución fija (típicamente 128x128 o 256x256 píxeles)
2. La red procesa esa imagen y produce:
   -  Un mapa de "hay una cara acá" (heatmap).
   -  Coordenadas de la caja.
   -  Coordenadas de los 6 keypoints.
3. Se aplica un filtro de supresión de no-máximos para eliminar cajas duplicadas.
4. Se descartan las detecciones con confianza menor al umbral.

BlazeFace tiene dos variantes:
- **Short range**: para caras cercanas a la cámara
- **Full range** para caras lejanas

## 5. Requisitos e instalación
**Requisitos**
- Python 3.9 o superior
- Una webcam

**Dependencias**
```bash
pip install mediapipe opencv-python
```

**Decargar el modelo (en caso de no tenerlo)**
1. Ir a https://developers.google.com/edge/mediapipe/solutions/vision/face_detector 
2. Buscar la sección **Models**
3. Decargar `blaze_face_short_range.tflite`
4. Ponerlo en la misma carpeta que el script

## 6. Ejecutar el código
Desde la terminal, en la carpeta del proyecto:
``` bash
py 01_face_detection.py
```

**Controles**
- `q` o `ESC` para cerrar la aplicación