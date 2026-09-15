# Guía Puntos Claves del Rostro (Face Landmarks) con MediaPipe Tasks

Esta guía acompaña al script `02_face_landmarks.py`. Explica qué son los
puntos faciales, cómo funciona el modelo, qué devuelve y cómo ejecutarlo.

---

## Índice
- [Guía Puntos Claves del Rostro (Face Landmarks) con MediaPipe Tasks](#guía-puntos-claves-del-rostro-face-landmarks-con-mediapipe-tasks)
  - [Índice](#índice)
  - [1. ¿Qué son los landmarks?](#1-qué-son-los-landmarks)
  - [2. Diferencia con la detección de rostros](#2-diferencia-con-la-detección-de-rostros)
  - [3. ¿Qué devuelve el modelo?](#3-qué-devuelve-el-modelo)
  - [4. Cómo funciona el modelo por dentro](#4-cómo-funciona-el-modelo-por-dentro)
  - [5. Requisitos e instalación](#5-requisitos-e-instalación)
  - [6. Ejecutar el código](#6-ejecutar-el-código)

---

## 1. ¿Qué son los landmarks?
Los **face landmarks** son coordenadas específicas sobre la cara: la  punta de la nariz, el borde de los labios, el contorno de los ojos, etc. 

MediaPipe Facec Landmarker detecta **478 puntos en 3D** sobre el rostro, formando una malla que cubre toda la superficie de la cara. 

---

## 2. Diferencia con la detección de rostros
La detección de rostros dice *dónde* está la cara, mientras que los puntos faciales dicen *cómo es* la cara. Este modelo tiene más precisión que solamente la detección, a costa de velocidad

---

## 3. ¿Qué devuelve el modelo?
El `FaceLandmarkerResult` contiene:
- **Face Landmarks**
    Una lista de mallas faciales, cada una con 478 puntos 3D normalizados
    - `x`: ancho de la imagen
    - `y`: alto de la imagen
    - `z`: profundidad relativa
- **Face Blendshapes**
    Una lista de coeficientes (entre 0.0 y 1.0) que representan expresiones faciales:

    **Cejas (Brow)**
    - `browDownLeft / browDownRight`: Ceño fruncido
    - `browInnerUp`: Ceja interna hacia arriba (sorpresa, preocupación)
    - `browOuterUpLeft / browOuterUpRight`: Ceja externa hacia arriba

    **Ojos (Eye)**
    - `eyeBlinkLeft / eyeBlinkRight`: Parpadeo
    - `eyeLookDownLeft / eyeLookDownRight`: Mirada hacia abajo
    - `eyeLookInLeft / eyeLookInRight`: Mirada hacia adentro
    - `eyeLookOutLeft / eyeLookOutRight`: Mirada hacia afuera
    - `eyeLookUpLeft / eyeLookUpRight`: Mirada hacia arriba
    - `eyeSquintLeft / eyeSquintRight`: Ojos entornados
    - `eyeWideLeft / eyeWideRight`: Ojos muy abiertos

    **Cachetes (Cheek)**
    - `cheekPuff`: Cachetes infladas
    - `cheekSquintLeft / cheekSquintRight`: Cachetes tensas

    **Mandíbula (Jaw)**
    - `jawForward`: Mandíbula hacia adelante
    - `jawLeft / jawRight`: Mandíbula hacia los costados
    - `jawOpen`: Boca abierta

    **Boca (Mouth)**
    - `mouthClose`: Boca cerrada
    - `mouthDimpleLeft / mouthDimpleRight`: Hoyuelos
    - `mouthFrownLeft / mouthFrownRight`: Comisuras hacia abajo
    - `mouthFunnel`: Boca en forma de embudo
    - `mouthLeft / mouthRight`: Boca hacia los costados
    - `mouthLowerDownLeft / mouthLowerDownRight`: Labio inferior hacia abajo
    - `mouthPressLeft / mouthPressRight`: Labios apretados
    - `mouthPucker`: Boca fruncida
    - `mouthRollLower / mouthRollUpper`: Labios hacia adentro
    - `mouthShrugLower / mouthShrugUpper`: Labios elevados
    - `mouthSmileLeft / mouthSmileRight`: Sonrisa
    - `mouthStretchLeft / mouthStretchRight`: Comisuras estiradas
    - `mouthUpperUpLeft / mouthUpperUpRight`: Labio superior hacia arriba

    **Nariz (Nose)**
    - `noseSneerLeft / noseSneerRight`: Nariz arrugada

    **Neutral**
    - `_neutral`: Expresión neutra, se usa para calibrar

---

## 4. Cómo funciona el modelo por dentro
Face Landmarker usa una cascada de 3 modelos:
1. **Detección del rostro**: un modelo encuentra caras en la imagen, similar a BlazeFace
2. **FaceMesh**: recibe la región del rostro y produce los puntos 3D
3. **Blendshapes**: toma los puntos y predice 52 coeficientes que representan la expresión

---

## 5. Requisitos e instalación
**Requisitos**
- Python 3.9 o superior
- Una webcam
- MediaPipe y OpenCV

**Instalación**
```bash
pip install mediapipe opencv-python
```

**Descargar el modelo**
1. Ir a https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
2. Buscar la sección **Models**
3. Decargar `blaze_face_short_range.tflite`
4. Ponerlo en la misma carpeta que el script
---
## 6. Ejecutar el código
Desde la terminal, en la carpeta del proyecto:
``` bash
py 02_face_landmarks.py
```
**Controles**
- `q` o `ESC` para cerrar la aplicación