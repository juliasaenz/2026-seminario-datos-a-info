# Guía: Reconocimiento de Gestos (Hand Gesture Recognition) con MediaPipe Tasks

Esta guía acompaña al script `04_hand_gestures.py`. Explica qué es el
reconocimiento de gestos, cómo funciona el modelo, qué devuelve y cómo
ejecutarlo.

---

## Índice

- [Guía: Reconocimiento de Gestos (Hand Gesture Recognition) con MediaPipe Tasks](#guía-reconocimiento-de-gestos-hand-gesture-recognition-con-mediapipe-tasks)
  - [Índice](#índice)
  - [1. ¿Qué es el reconocimiento de gestos?](#1-qué-es-el-reconocimiento-de-gestos)
  - [2. ¿Qué devuelve el modelo?](#2-qué-devuelve-el-modelo)
  - [3. Cómo funciona el modelo por dentro](#3-cómo-funciona-el-modelo-por-dentro)
  - [4. Requisitos e instalación](#4-requisitos-e-instalación)
    - [Requisitos](#requisitos)
    - [Instalación](#instalación)
  - [5. Ejecutar el código](#5-ejecutar-el-código)

---

## 1. ¿Qué es el reconocimiento de gestos?

El **reconocimiento de gestos** (gesture recognition) es la tarea de
identificar QUÉ está haciendo una mano, no solo DÓNDE está.

A diferencia de Hand Landmarks (que solo devuelve los 21 puntos), el
reconocimiento de gestos agrega una **clasificación**: "esta mano está
haciendo un puño cerrado", "esta mano está mostrando la palma abierta",
etc.

MediaPipe Gesture Recognizer reconoce **8 gestos predefinidos**, que
cubren las señas más comunes:
- `None`: Sin gesto reconocido
- `Closed_Fist`: Puño cerrado
- `Open_Palm`: Palma abierta
- `Pointing_Up`: Dedo índice apuntando hacia arriba
- `Thumb_Down`: Pulgar hacia abajo
- `Thumb_Up`: Pulgar hacia arriba
- `Victory`: Seña de victoria (dos dedos)
- `ILoveYou`: Seña "te amo" (lengua de señas)

---

## 2. ¿Qué devuelve el modelo?

El `GestureRecognizerResult` contiene:
- **gestures**
    Una lista de clasificaciones de gestos. Por cada mano detectada, hay una lista de categorías (normalmente una sola) con:
  - `category_name`: el nombre del gesto (`"Thumb_Up"`, `"Victory"`, etc.)
  - `score`: la confianza (0.0 a 1.0)

    **Nota importante**: el índice del gesto siempre es -1, porque los índices de múltiples clasificadores no se pueden consolidar en un índice significativo. Para acceder al gesto, se usa `gestos[0]`
- **hand_landmarks**
    Los mismos 21 puntos que devuelve Hand Landmarker. El reconocimiento de gestos usa internamente estos puntos para clasificar
- **handedness**
    La lateralidad de cada mano: `"Left"` o `"Right"`.

---

## 3. Cómo funciona el modelo por dentro

Gesture Recognizer usa un **modelo bundle** que contiene varios modelos
en cadena:
1. **Detección de palma** (Palm Detection): Primero, un modelo detector localiza las palmas en la imagen. Es el mismo BlazePalm que usa Hand Landmarker
2. **Puntos de la mano** (Hand Landmarks): El segundo modelo extrae los **21 puntos 3D** de la mano
3. **Clasificación de gestos**: El tercer modelo toma los 21 puntos y clasifica el gesto

---

## 4. Requisitos e instalación

### Requisitos

- Python 3.9 o superior.
- Una webcam.
- MediaPipe y OpenCV.

### Instalación

```bash
pip install mediapipe opencv-python
```

**Descargar el modelo**
1. Ir a: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer
2. Buscar la sección Models
3. Descargar `gesture_recognizer.task`
4. Colocarlo en la misma carpeta que el script

---

## 5. Ejecutar el código
Desde la terminal, en la carpeta del proyecto:
``` bash
py 04_hand_gestures.py
```
**Controles**
- `q` o `ESC` para cerrar la aplicación