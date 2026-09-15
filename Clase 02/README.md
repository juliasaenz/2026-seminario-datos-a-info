# Ejemplos de MediaPipe

Este paquete contiene archivos de códigos independientes, uno por cada modelo de MediaPipe. Todos usan la cámara web y están pensados para que los y las estudiantes los editen, rompan y experimenten.

---

## Índice
- [Ejemplos de MediaPipe](#ejemplos-de-mediapipe)
  - [Índice](#índice)
  - [1. ¿Qué es MediaPipe?](#1-qué-es-mediapipe)
  - [2. Estructura de las carpetas](#2-estructura-de-las-carpetas)

---

## 1. ¿Qué es MediaPipe?

MediaPipe es una librería de Google que ofrece modelos de *machine learning* ya entrenados y optimizados para correr en tiempo real, sin necesidad de entrenar nada. Cada modelo recibe una imagen (un frame de la cámara) y devuelve información estructurada: coordenadas de puntos clave, máscaras, clasificaciones, etc. Esa información es la "materia prima" que después podemos usar para generar arte: dibujar, controlar sonido, mover partículas, disparar video, etc.

---

## 2. Estructura de las carpetas

Cada carpeta tiene un código de ejemplo usando alguno de los modelos de MediaPipe, junto con su propia guía en el `README.md`. Los modelos con ejemplos son:
1. [Face Detection](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector) (Detección de rostros)
2. [Face Landmarks](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) (Puntos clave de  rostros)
3. [Hand Landmarks](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) (Puntos claves de manos)
4. [Gesture Recognition](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer) (Reconocimiento de gestos)
5. [Pose Landmarks](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) (Puntos claves de pose)
6. [Holistic Landmarks](https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker) (Puntos claves de rosto + manos + pose)
7. [Object Detection](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector) (Detección de objetos)

Para más información sobre cada uno, ver el ejemplo en la carpeta o la documentación de MediaPipe