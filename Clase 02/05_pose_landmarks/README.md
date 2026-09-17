# Guía: Puntos de la Pose (Pose Landmarks) con MediaPipe Tasks

Esta guía acompaña al script `05_pose_landmarks.py`. Explica qué son los
puntos de la pose, cómo funciona el modelo, qué devuelve y cómo ejecutarlo.

---

## Índice

- [Guía: Puntos de la Pose (Pose Landmarks) con MediaPipe Tasks](#guía-puntos-de-la-pose-pose-landmarks-con-mediapipe-tasks)
  - [Índice](#índice)
  - [1. ¿Qué son los puntos de la pose?](#1-qué-son-los-puntos-de-la-pose)
    - [Los 33 puntos clave](#los-33-puntos-clave)
  - [2. ¿Qué devuelve el modelo?](#2-qué-devuelve-el-modelo)
  - [3. Cómo funciona el modelo por dentro](#3-cómo-funciona-el-modelo-por-dentro)
    - [3.1. Variantes del modelo](#31-variantes-del-modelo)
    - [3.2. Basado en BlazePose](#32-basado-en-blazepose)
  - [4. Requisitos e instalación](#4-requisitos-e-instalación)
  - [5. Cómo ejecutar el código](#5-cómo-ejecutar-el-código)

---

## 1. ¿Qué son los puntos de la pose?

Los **puntos de la pose** (pose landmarks) son coordenadas específicas
sobre las articulaciones y partes principales del cuerpo humano

MediaPipe Pose Landmarker detecta **33 puntos clave en 3D** que forman un
esqueleto completo del cuerpo
- Rastrear la posición del cuerpo entero
- Analizar posturas y movimientos
- Detectar gestos con brazos y piernas

### Los 33 puntos clave

MediaPipe numera los puntos del 0 al 32 :
- 0 | Nariz 
- 1-6 | Ojos (internos, centros, externos) 
- 7-8 | Orejas 
- 9-10 | Boca (esquinas) 
- 11-12 | Hombros 
- 13-14 | Codos 
- 15-16 | Muñecas 
- 17-22 | Manos (meñique, índice, pulgar) 
- 23-24 | Caderas 
- 25-26 | Rodilla 
- 27-28 | Tobillos 
- 29-30 | Talones 
- 31-32 | Dedos de los pies 

---

## 2. ¿Qué devuelve el modelo?

El `PoseLandmarkerResult` contiene:
- **pose_landmarks**
    Una lista de esqueletos de poses. Cada esqueleto es una lista de **33 puntos 3D** con coordenadas normalizadas:
  - `x`: 0.0 a 1.0 (ancho de la imagen)
  - `y`: 0.0 a 1.0 (alto de la imagen)
  - `z`: profundidad relativa (0.0 = punto de referencia)
  - `visibility`: 0.0 a 1.0 (probabilidad de que el punto sea visible)
  - `presence`: 0.0 a 1.0 (probabilidad de que el punto esté presente)

---

## 3. Cómo funciona el modelo por dentro

Pose Landmarker usa un **modelo bundle** que contiene dos modelos en
cadena :
1. **Detección de pose** (Pose Detection): un modelo detector localiza la presencia de cuerpos humanos en la imagen y devuelve una región aproximada. Este modelo se ejecuta solo cuando es necesario (al inicio, o cuando se pierde el seguimiento)
2. **Puntos de la pose** (Pose Landmarks): el segundo modelo recibe la región detectada y produce los 33 puntos 3D del cuerpo completo. Este modelo se ejecuta en cada frame

### 3.1. Variantes del modelo

El bundle viene en tres tamaños :
- **Lite**: Muy rápida
- **Full**: Balance velocidad-calidad
- **Heavy**: Lenta, mejor calidad

### 3.2. Basado en BlazePose

El modelo usa la arquitectura **BlazePose**, que a su vez usa **GHUM**,
un pipeline de modelado 3D de forma humana. Esto permite estimar la
pose 3D completa de una persona en una imagen o video .

---

## 4. Requisitos e instalación

**Requisitos**

- Python 3.9 o superior.
- Una webcam.
- MediaPipe y OpenCV.

**Instalación**

```bash
pip install mediapipe opencv-python
```

**Descargar el modelo**

1. Ir a: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
2. Buscar la sección **Models**.
3. Descargar `pose_landmarker_lite.task`.
4. Colocarlo en la misma carpeta que el script.

---

## 5. Cómo ejecutar el código

```bash
python 05_pose_landmarks.py
```
