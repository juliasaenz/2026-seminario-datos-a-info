# Guía: Puntos de la Mano (Hand Landmarks) con MediaPipe Tasks

Esta guía acompaña al script `03_hand_landmarks.py`. Explica qué son los
puntos de la mano, cómo funciona el modelo, qué devuelve y cómo ejecutarlo.

---

## Índice

- [Guía: Puntos de la Mano (Hand Landmarks) con MediaPipe Tasks](#guía-puntos-de-la-mano-hand-landmarks-con-mediapipe-tasks)
  - [Índice](#índice)
  - [1. ¿Qué son los puntos de la mano?](#1-qué-son-los-puntos-de-la-mano)
    - [1.1. Los 21 puntos clave](#11-los-21-puntos-clave)
  - [2. ¿Qué devuelve el modelo?](#2-qué-devuelve-el-modelo)
  - [3. Cómo funciona el modelo por dentro](#3-cómo-funciona-el-modelo-por-dentro)
  - [4. Requisitos e instalación](#4-requisitos-e-instalación)
  - [5. Ejecutar el código](#5-ejecutar-el-código)

---

## 1. ¿Qué son los puntos de la mano?

Los **puntos de la mano** (hand landmarks) son coordenadas específicas
sobre las articulaciones y la punta de cada dedo.

MediaPipe Hand Landmarker detecta **21 puntos clave en 3D** por cada mano,
formando un esqueleto que permite rastrear la posición y forma de la mano
en tiempo real.

Estos puntos permiten:
- Rastrear la posición y orientación de la mano
- Detectar la posición de cada dedo
- Reconocer gestos (puño, palma abierta, victoria, etc.)

### 1.1. Los 21 puntos clave

MediaPipe numera los puntos del 0 al 20:
- 0: Muñeca (wrist)
- 1-4: Pulgar (thumb) - de la base a la punta
- 5-8: Índice (index) - de la base a la punta
- 9-12: Medio (middle) - de la base a la punta
- 13-16: Anular (ring) - de la base a la punta
- 17-20: Meñique (pinky) - de la base a la punta

## 2. ¿Qué devuelve el modelo?
El `HandLandmarkerResult` tiene:
- **hand_landmarks**
    Una lista de esqueletos de manos, cada uno con la lista de 21 puntos con coordenadas normalizadas (x, y, z)
- **handedness**
    Una lista de clasificaciones de lateralidad

## 3. Cómo funciona el modelo por dentro
Hand Landmarker usa un modelo bundle que contiene dos modelos:
1. **Detección de palma**: un modelo llamado BlazePalm localiza las palmas en la imagen completa. Se ejecuta solo cuando es necesario (cuando no ve ninguna mano)
2. **Puntos de la mano**: recibe la región de la palma detectada y produce los 21 puntos de la mano. Se ejecuta en cada frame

## 4. Requisitos e instalación
**Requisitos**
- Python 3.9 o superior
- Una webcam
- MediaPipe y openCV

**Instalación**
```bash
pip install mediapipe opencv-python
```

**Descargar el modelo**
1. Ir a: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
2. Buscar la sección Models
3. Descargar `hand_landmarker.task`
4. Colocarlo en la misma carpeta que el script

## 5. Ejecutar el código
Desde la terminal, en la carpeta del proyecto:
``` bash
py 03_hand_landmarks.py
```
**Controles**
- `q` o `ESC` para cerrar la aplicación