# Guía: Landmarks Holísticos (Holistic Landmarker) con MediaPipe Tasks

Esta guía acompaña al script `06_holistic_landmarks.py`. Explica qué es el detector holístico, cómo funciona, qué devuelve y cómo ejecutarlo.

---

## Índice

- [Guía: Landmarks Holísticos (Holistic Landmarker) con MediaPipe Tasks](#guía-landmarks-holísticos-holistic-landmarker-con-mediapipe-tasks)
  - [Índice](#índice)
  - [1. ¿Qué es el Holistic Landmarker?](#1-qué-es-el-holistic-landmarker)
  - [2. ¿Qué devuelve el modelo?](#2-qué-devuelve-el-modelo)
  - [3. Cómo funciona el modelo por dentro](#3-cómo-funciona-el-modelo-por-dentro)
    - [3.1. Seguimiento entre frames](#31-seguimiento-entre-frames)
  - [Para optimizar, el pipeline usa la pose como **guía de seguimiento**. Si el cuerpo se mueve, las ROIs se actualizan automáticamente. Esto evita ejecutar los detectores de rostro y manos en cada frame](#para-optimizar-el-pipeline-usa-la-pose-como-guía-de-seguimiento-si-el-cuerpo-se-mueve-las-rois-se-actualizan-automáticamente-esto-evita-ejecutar-los-detectores-de-rostro-y-manos-en-cada-frame)
  - [4. Requisitos e instalación](#4-requisitos-e-instalación)
  - [5. Ejecutar el código](#5-ejecutar-el-código)

---

## 1. ¿Qué es el Holistic Landmarker?

El **Holistic Landmarker** (detector holístico) combina en un solo modelo la detección de pose, rostro y manos. Es como ejecutar los scripts 02, 03 y 05 al mismo tiempo, pero con un único modelo bundle.

Detecta un total de **553 landmarks** en tiempo real [citation:4]:

- Pose (cuerpo) | 33 puntos
- Rostro (face mesh) | 478 puntos
- Manos (21 × 2) | 42 puntos
- **Total** | **553** puntos

Esto permite analizar el cuerpo completo, incluyendo expresiones faciales
y gestos con las manos, todo en un solo paso

---

## 2. ¿Qué devuelve el modelo?

El `HolisticLandmarkerResult` contiene:
- **pose_landmarks**
    Los 33 puntos de la pose (igual que Pose Landmarker). Coordenadas normalizadas (0.0 a 1.0).
- **face_landmarks**
    Los 478 puntos de la malla facial (igual que Face Landmarker). Incluye los 10 puntos del iris.
- **left_hand_landmarks / right_hand_landmarks**
    Los 21 puntos de cada mano (igual que Hand Landmarker). Vienen separados en listas distintas para mano izquierda y derecha.
- **pose_world_landmarks** (opcional)
    Coordenadas de los 33 puntos de pose en espacio real (metros aproximados), relativas al centro de la cadera.
- **face_blendshapes** (opcional)
    Los 52 coeficientes de expresión facial. Se activan con `output_face_blendshapes=True`.
- **segmentation_mask** (opcional)
    Máscara de segmentación de la persona. Se activa con `output_segmentation_mask=True`.

---

## 3. Cómo funciona el modelo por dentro

Holistic Landmarker usa un **pipeline de múltiples etapas**:
1. **Detección de pose**: Primero, un detector de pose (BlazePose) localiza el cuerpo humano en la imagen y devuelve los 33 puntos clave
2. **Derivación de regiones de interés** (ROI): Usando los puntos de la pose, el pipeline deriva automáticamente las regiones donde deberían estar el rostro y las manos. Por ejemplo:
   - Los puntos 7 y 8 (orejas) ayudan a ubicar el rostro
   - Los puntos 15 y 16 (muñecas) ayudan a ubicar las manos
3. **Re-crop de precisión**: Como la pose se estima en baja resolución, las regiones derivadas no son precisas. Un modelo ligero de **re-crop** (re-corte) refina cada ROI antes de pasarla a los modelos específicos. Esto cuesta solo ~10% extra de tiempo
4. ** Modelos específicos**:
    - **Face Landmarker**: procesa la ROI del rostro → 478 puntos.
   - **Hand Landmarker**: procesa cada ROI de mano → 21 puntos por mano.
5. **Fusión**: Todos los landmarks se juntan en un único resultado con 553 puntos

### 3.1. Seguimiento entre frames

Para optimizar, el pipeline usa la pose como **guía de seguimiento**. Si el cuerpo se mueve, las ROIs se actualizan automáticamente. Esto evita ejecutar los detectores de rostro y manos en cada frame
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
1. Ir a: https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker
2. Buscar la sección Models
3. Descargar `holistic_landmarker.task`
4. Ponerlo en la misma carpeta que el script

---

## 5. Ejecutar el código
Desde la terminal, en la carpeta del proyecto:
``` bash
py 06_hollistic_landmarks.py
```
**Controles**
- `q` o `ESC` para cerrar la aplicación