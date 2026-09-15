# Ejemplos de MediaPipe

Este paquete contiene archivos de códigos independientes, uno por cada modelo de MediaPipe. Todos usan la cámara web y están pensados para que los y las estudiantes los editen, rompan y experimenten.

## 1. ¿Qué es MediaPipe?

MediaPipe es una librería de Google que ofrece modelos de *machine learning* ya entrenados y optimizados para correr en tiempo real, sin necesidad de entrenar nada. Cada modelo recibe una imagen (un frame de la cámara) y devuelve información estructurada: coordenadas de puntos clave, máscaras, clasificaciones, etc. Esa información es la "materia prima" que después podemos usar para generar arte: dibujar, controlar sonido, mover partículas, disparar video, etc.

## 2. Estructura

### 2.1. Estructura de la carpeta

```
clase 02/
├── README.md                      <- este archivo
├── requirements.txt                <- dependencias a instalar
├── 01_deteccion_rostros.py         <- Face Detection
├── 02_malla_facial.py              <- Face Landmarker (Face Mesh)
├── 03_manos.py                     <- Hand Landmarker
├── 04_pose_corporal.py             <- Pose Landmarker
├── 05_holistico.py                 <- Holistic (cara + manos + cuerpo)
├── 06_segmentacion_fondo.py        <- Selfie Segmentation
└── 07_reconocimiento_gestos.py     <- Gesture Recognizer
```
### 2.2. Estructura de los archivos

Cada script tiene la misma estructura para que sea fácil moverse entre ellos:

1. **Encabezado** con una explicación de qué hace el modelo y una explicación de cómo funciona por dentro.
2. **Bloque de PARÁMETROS EDITABLES**  separado del resto del código, para que se pueda experimentar.
3. **Bucle principal** que lee la cámara, procesa cada frame y dibuja el resultado.
4. Una **zona de experimentación** (marcada con ✏️) con una idea de proyecto
   ya armada, para que sirva de punto de partida.

## 3. Instalación

Se recomienda usar un entorno virtual (`venv` o `conda`) con **Python 3.9 a 3.12**.

```bash
pip install -r requirements.txt
```

Esto instala `mediapipe`, `opencv-python` y `numpy`.

> ⚠️ El script `07_reconocimiento_gestos.py` necesita descargar un archivo de
> modelo (`gesture_recognizer.task`, ~8 MB) la primera vez que se ejecuta. El
> propio script lo descarga automáticamente si hay conexión a internet.

### 3.1. Cómo correr cada script

```bash
python 01_deteccion_rostros.py
```

Se abre una ventana con la imagen de la cámara y las anotaciones dibujadas
encima. Para cerrarla, hay que tener la ventana seleccionada y presionar la
tecla **`q`** (o `ESC`).