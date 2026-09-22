# Ejemplos de MediaPipe

Este paquete contiene archivos de código independientes, uno por cada
modelo de MediaPipe. Todos usan la cámara web y están pensados para que
los y las estudiantes los editen, rompan y experimenten.

Cada carpeta tiene:
- Un archivo `.py` con el código del modelo.
- Un `README.md` con la guía específica de ese modelo.
- El archivo del modelo (`.task` o `.tflite`) que el código necesita.

---

## Índice

- [Ejemplos de MediaPipe](#ejemplos-de-mediapipe)
  - [Índice](#índice)
  - [1. ¿Qué es MediaPipe?](#1-qué-es-mediapipe)
  - [2. Estructura del paquete](#2-estructura-del-paquete)
  - [3. Instalación desde cero](#3-instalación-desde-cero)
    - [3.1. Windows](#31-windows)
    - [3.2. macOS](#32-macos)
    - [3.3. Verificar que quedó todo bien](#33-verificar-que-quedó-todo-bien)
    - [3.4. Problemas comunes de instalación](#34-problemas-comunes-de-instalación)
  - [4. Cómo ejecutar los ejemplos](#4-cómo-ejecutar-los-ejemplos)
    - [4.1. Instalar VS Code](#41-instalar-vs-code)
    - [4.2. Instalar la extensión de Python en VS Code](#42-instalar-la-extensión-de-python-en-vs-code)
    - [4.3. Abrir la carpeta del proyecto](#43-abrir-la-carpeta-del-proyecto)
    - [4.4. Ejecutar un ejemplo](#44-ejecutar-un-ejemplo)
    - [4.5. Cerrar el ejemplo](#45-cerrar-el-ejemplo)
    - [4.6. La primera vez que ejecutás un script](#46-la-primera-vez-que-ejecutás-un-script)
    - [4.7. Controles comunes a todos los scripts](#47-controles-comunes-a-todos-los-scripts)
    - [4.8. Problemas comunes](#48-problemas-comunes)
  - [5. Las dos APIs de MediaPipe](#5-las-dos-apis-de-mediapipe)
    - [5.1. API clásica: mp.solutions](#51-api-clásica-mpsolutions)
    - [5.2. API moderna: mediapipe.tasks](#52-api-moderna-mediapipetasks)
  - [6. Cómo leer las coordenadas](#6-cómo-leer-las-coordenadas)
  - [7. Tabla comparativa de los modelos](#7-tabla-comparativa-de-los-modelos)
  - [8. Convenciones de todos los scripts](#8-convenciones-de-todos-los-scripts)
    - [8.1. Estructura del archivo](#81-estructura-del-archivo)
    - [8.2. Los dos bloques de parámetros editables](#82-los-dos-bloques-de-parámetros-editables)
    - [8.3. Toggles comunes](#83-toggles-comunes)
    - [8.4. La pausa](#84-la-pausa)
    - [8.5. El panel de información](#85-el-panel-de-información)
    - [8.6. El panel de detalle (derecha)](#86-el-panel-de-detalle-derecha)
    - [8.7. Los timestamps](#87-los-timestamps)
    - [8.8. Cómo se dibujan los landmarks](#88-cómo-se-dibujan-los-landmarks)
    - [8.9. Orden de las operaciones en el loop](#89-orden-de-las-operaciones-en-el-loop)

---

## 1. ¿Qué es MediaPipe?

MediaPipe es una librería de Google que ofrece modelos de *machine learning*
ya entrenados y optimizados para correr en tiempo real, sin necesidad de
entrenar nada.

Cada modelo recibe una imagen (un frame de la cámara) y devuelve
información estructurada: coordenadas de puntos clave, cajas, máscaras,
clasificaciones, etc. Esa información es la "materia prima" que después
podemos usar para generar arte: dibujar, controlar sonido, mover
partículas, disparar video, etc.

MediaPipe corre localmente en tu computadora. No manda nada a internet.
La cámara captura, el modelo procesa, y vos decidís qué hacer con el
resultado. Eso lo hace ideal para instalaciones artísticas que necesitan
funcionar sin conexión.

---

## 2. Estructura del paquete

Cada modelo vive en su propia carpeta, con esta estructura:

```Ejemplos-MediaPipe/
├── README.md <- este archivo
├── 01_face_detection/
│ ├── 01_face_detection.py
│ ├── blaze_face_short_range.tflite
│ └── README.md
├── 02_face_landmarks/
│ ├── 02_face_landmarks.py
│ ├── face_landmarker.task
│ └── README.md
├── 03_hand_landmarks/
│ ├── 03_hand_landmarks.py
│ ├── hand_landmarker.task
│ └── README.md
├── 04_hand_gestures/
│ ├── 04_hand_gestures.py
│ ├── gesture_recognizer.task
│ └── README.md
├── 05_pose_landmarks/
│ ├── 05_pose_landmarks.py
│ ├── pose_landmarker_lite.task
│ └── README.md
├── 06_holistic_landmarks/
│ ├── 06_holistic_landmarks.py
│ ├── holistic_landmarker.task
│ └── README.md
```


**Importante**: los archivos de modelo ya vienen incluidos en cada
carpeta. No hace falta descargarlos por separado.

---

## 3. Instalación desde cero

Esta sección asume que **nunca instalaste Python** y que **no estás
familiarizado con la consola**. Si ya tenés Python y sabés usar la
terminal, podés saltar directo al paso 3.3.

### 3.1. Windows

**Paso 1: Descargar Python**

1. Ir a https://www.python.org/downloads/
2. Hacer clic en el botón amarillo grande que dice "Download Python 3.12.x".
3. Se descarga un archivo `.exe`. Doble clic para ejecutarlo.

**Paso 2: Instalar Python**

Muy importante: en la primera pantalla del instalador, **tildar la casilla
que dice "Add Python to PATH"** (agregar Python al PATH). Si no tildás
eso, la consola no va a encontrar Python después y vas a tener que
reinstalar.

Después hacer clic en "Install Now" y esperar a que termine.

**Paso 3: Abrir la consola**

En Windows, "Terminal" o "PowerShell".
Podés abrirla así:

- Apretar la tecla Windows.
- Escribir `cmd` o `powershell`.
- Apretar Enter.

Se abre una ventana negra con texto. Eso es la consola.

**Paso 4: Verificar que Python está instalado**

En la consola, escribir:

```bash
python --version
```

Debería responder algo como Python 3.12.4. Si dice eso, Python está
listo.

**Paso 5: Instalar MediaPipe y OpenCV**

En la misma consola, escribir:

```bash
pip install mediapipe opencv-python
```

Paso 6: Descargar el paquete de ejemplos

Descargá la carpeta Ejemplos-MediaPipe completa (la que contiene este
README) y guardala.

### 3.2. macOS

**Paso 1: Abrir la Terminal**

La Terminal está en `Aplicaciones > Utilidades > Terminal`. También podés
buscarla con Spotlight (Cmd + Espacio) y escribir "Terminal".

**Paso 2: Verificar si ya tenés Python**

En la Terminal, escribir:
```bash

python3 --version
```

Si responde algo como Python 3.12.x, ya lo tenés y podés saltar al
paso 4. Si dice "command not found" o algo parecido, seguí al paso 3.

**Paso 3: Instalar Python**

La forma más simple es bajar el instalador oficial:

1. Ir a https://www.python.org/downloads/
2. Hacer clic en el botón amarillo grande que dice "Download Python 3.12.x".
3. Se descarga un archivo .pkg. Doble clic y seguir los pasos del
    instalador (todo "Siguiente", aceptar términos, etc.).

**Paso 4: Instalar MediaPipe y OpenCV**

En la Terminal, escribir:
```bash

pip3 install mediapipe opencv-python
```

**Paso 5: Descargar el paquete de ejemplos**

Descargá la carpeta Ejemplos-MediaPipe completa y guardala en algún
lugar.

### 3.3. Verificar que quedó todo bien

Para asegurarte de que Python, MediaPipe y OpenCV están instalados, en
la consola escribí estos tres comandos, uno por uno:

```bash
python -c "import mediapipe; print(mediapipe.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

Si cada uno imprime un número de versión (por ejemplo 0.10.35 y
4.10.0), está todo listo.

Si estás en Mac y `python` no funciona, probá con `python3` en lugar de
python. En Mac, python3 es el nombre correcto.

### 3.4. Problemas comunes de instalación

**"python no se reconoce como un comando" (Windows)**

No tildaste la casilla "Add Python to PATH" durante la instalación.
Solución: desinstalar Python y volver a instalarlo, esta vez tildando esa
casilla.

**"pip no se reconoce como un comando" (Windows)**

Mismo problema. Reinstalar Python tildando "Add Python to PATH".

**"pip3: command not found" (Mac)**

Probá con `python3 -m pip install mediapipe opencv-python` en lugar de
`pip3 install` ....

**La instalación de MediaPipe falla (Mac con chip M1/M2/M3)**

A veces pasa si tenés un Python viejo compilado para Intel. Verificá con:
```bash

python3 -c "import platform; print(platform.machine())"
```

Si dice `x86_64` y tu Mac es Apple Silicon, tenés un Python "equivocado".
Descargá un Python nuevo desde python.org (que va a ser arm64 nativo) y
reinstalá MediaPipe.

**La cámara no abre**

Otra aplicación la está usando (Zoom, Teams, Meet, etc.). Cerrá todo y
volvé a intentar. Si no funciona, en el script probá cambiar
`CAMARA_INDEX = 0` por `CAMARA_INDEX = 1`.

---

## 4. Cómo ejecutar los ejemplos

Todos los ejemplos se ejecutan desde **Visual Studio Code** (VS Code), un
editor de código gratuito que corre en Windows, Mac y Linux. La ventaja
de usarlo es que no hace falta escribir comandos en la consola para
navegar entre carpetas: se abre la carpeta del proyecto y se ejecuta el
archivo con un botón.

### 4.1. Instalar VS Code

**Windows:**

1. Ir a https://code.visualstudio.com/
2. Hacer clic en el botón que dice "Download for Windows".
3. Se descarga un archivo `.exe`. Doble clic para ejecutarlo.
4. Aceptar los términos y hacer clic en "Siguiente" hasta que termine.

**macOS:**

1. Ir a https://code.visualstudio.com/
2. Hacer clic en el botón que dice "Download for macOS".
3. Se descarga un archivo `.zip`. Doble clic para descomprimirlo.
4. Se extrae una aplicación llamada "Visual Studio Code". Arrastrala a la carpeta `Aplicaciones`.

### 4.2. Instalar la extensión de Python en VS Code

La extensión de Python le dice a VS Code cómo ejecutar archivos `.py`.

1. Abrir VS Code.
2. En la barra lateral izquierda, hacer clic en el ícono de **Extensiones** (son cuatro cuadraditos, uno de ellos despegado).
3. En la barra de búsqueda, escribir `Python`.
4. El primer resultado va a ser "Python" de Microsoft (tiene un tilde azul de verificado). Hacer clic en **Install**.

### 4.3. Abrir la carpeta del proyecto

1. Abrir VS Code.
2. En el menú superior, ir a **File > Open Folder...**
   (en Mac: **File > Open...**).
3. Abrir la carpeta en la que están los ejemplos

En la barra lateral izquierda de VS Code ahora vas a ver todas las
carpetas del proyecto: `01_face_detection`, `02_face_landmarks`, etc.

### 4.4. Ejecutar un ejemplo

1. En la barra lateral izquierda, hacer clic en la carpeta del ejemplo
   que querés probar, por ejemplo `01_face_detection`.
2. Hacer clic en el archivo `.py` que está adentro
   (`01_face_detection.py`). Se abre en el editor.
3. Arriba a la derecha del editor hay un botón con forma de **triángulo
   de "play"** (▶). Hacer clic ahí. También se puede apretar `F5` o
   `Ctrl+F5`.

VS Code abre una terminal integrada abajo, ejecuta el script, y después
de unos segundos se abre una ventana nueva con la imagen de tu webcam.

### 4.5. Cerrar el ejemplo

Para cerrar la ventana de la cámara, con la ventana activa:

- Apretar `q` o `ESC`.

La ventana se cierra y en la terminal de VS Code aparece el mensaje de
fin.

### 4.6. La primera vez que ejecutás un script

La primera vez puede aparecer una notificación de VS Code abajo a la
derecha que dice algo como:

> "We noticed a new environment has been created. Do you want to select
> it for the workspace folder?"

Seleccionar **Yes**. Esto le dice a VS Code que use el Python que
instalaste antes.

También puede aparecer:

> "Select Python Interpreter"

Hacer clic y elegir la versión de Python 3.12 que instalaste. Si hay
varias opciones, elegir la que diga "Recommended" o la que tenga el
número de versión más alto.

### 4.7. Controles comunes a todos los scripts

| Tecla | Acción |
|-------|--------|
| `q` / `ESC` | Cerrar el programa |
| `i` | Mostrar / ocultar el panel de información |
| `ESPACIO` | Pausar / reanudar la captura |
| Otras teclas | Varían según el script (ver el README de cada uno) |

### 4.8. Problemas comunes

**Al hacer clic en "play" no pasa nada**

Puede ser que VS Code no tenga seleccionado el intérprete de Python.
Abajo a la derecha del editor debería decir algo como `Python 3.12.x`.
Si no dice nada, hacer clic ahí y seleccionar la versión correcta.

**Aparece "ModuleNotFoundError: No module named 'mediapipe'"**

MediaPipe no está instalado en el Python que VS Code está usando.
Volver a la sección 3 y verificar que la instalación haya funcionado.
Si ya lo instalaste pero sigue fallando, puede ser que tengas varios
Python instalados y VS Code esté usando otro. Hacer clic en la versión
de Python abajo a la derecha del editor y elegir otra.

**La ventana de la cámara se abre pero está en negro**

Puede ser que otra aplicación esté usando la cámara (Zoom, Meet, Teams,
etc.). Cerrar todo y volver a ejecutar. Si sigue, en el script cambiar
`CAMARA_INDEX = 0` por `CAMARA_INDEX = 1` y volver a probar.

**En Mac aparece "Operation not permitted" al abrir la cámara**

macOS pide permiso explícito para usar la cámara. La primera vez que
ejecutes un script, va a aparecer un cartel pidiendo permiso. Hacer clic
en **Permitir**. Si lo rechazaste sin querer, ir a
**Configuración del Sistema > Privacidad y seguridad > Cámara** y
habilitar VS Code (o Terminal, según desde dónde ejecutes).

---

## 5. Las dos APIs de MediaPipe

MediaPipe tuvo (y todavía tiene) dos APIs. Entender la diferencia ayuda cuando busques tutoriales en internet, porque vas a encontrar ejemplos de las dos y no siempre aclaran cuál están usando.
### 5.1. API clásica: mp.solutions
```python

import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
resultado = hands.process(imagen_rgb)
```

- Más simple de escribir.
- Los modelos vienen incluidos en el paquete de MediaPipe.
- Estaba en modo "mantenimiento" desde hace años.
- Fue eliminada en MediaPipe 0.10.30 y versiones posteriores.

### 5.2. API moderna: mediapipe.tasks
```python

import mediapipe as mp
from mediapipe.tasks.python import vision

opciones = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="modelo.task"),
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.HandLandmarker.create_from_options(opciones)
resultado = detector.detect_for_video(imagen, timestamp)
```
- Requiere descargar el modelo aparte (un .task o .tflite).
- Es la API oficial y recomendada hoy.
- Permite cambiar el modelo según lo necesario (por ejemplo, "lite" vs "full" en pose).
- Es la que usan todos los scripts de este paquete.

Si encontrás un tutorial viejo que usa mp.solutions, está bien leerlo
para entender la lógica, pero el código no va a correr con la versión
actual de MediaPipe. Los ejemplos de este paquete usan la API Tasks, que
es la que sigue viva.

---
## 6. Cómo leer las coordenadas

Todos los modelos de MediaPipe devuelven las coordenadas de sus puntos
de una manera particular: normalizadas entre 0.0 y 1.0.

Eso significa que:
- `x = 0.0` es el borde izquierdo de la imagen.
- `x = 1.0` es el borde derecho.
- `y = 0.0` es el borde superior.
- `y = 1.0` es el borde inferior.

Una coordenada como `(0.5, 0.5)` está justo en el centro de la imagen,
sin importar el ancho ni el alto reales.

La ventaja de este sistema es que las coordenadas no dependen de la
resolución: si cambiás la cámara de 640x480 a 1280x720, las coordenadas
de un mismo punto siguen siendo las mismas. Eso hace que el código sea
más portable.

Para pasar de coordenadas normalizadas a píxeles (por ejemplo, para
dibujar sobre la imagen):
```python

x_px = int(x_normalizado * ancho_imagen)
y_px = int(y_normalizado * alto_imagen)
```

Algunos modelos devuelven además una coordenada `z` de profundidad
relativa. Un `z` más negativo suele significar "más cerca de la cámara".
La escala exacta depende del modelo, pero sirve para comparar puntos
entre sí (por ejemplo, saber qué mano está más adelante).

La mayoría de los modelos devuelve también un `score` o `confidence`
entre `0.0` y `1.0` que indica cuán seguro está el modelo de esa detección.
`0.9` o más suele ser "muy seguro"; `0.3` o menos suele ser ruido.

---
## 7. Tabla comparativa de los modelos

| # | Modelo | Qué devuelve | Cantidad | Velocidad |
|---|--------|--------------|----------|-----------|
| 01 | Face Detection | Caja + 6 keypoints + score | 1 caja por rostro | Muy rápida |
| 02 | Face Landmarks | 478 landmarks + 52 blendshapes | 478 puntos por rostro | Rápida |
| 03 | Hand Landmarks | 21 landmarks por mano + lateralidad | 21 por mano | Rápida |
| 04 | Gesture Recognition | 21 landmarks + gesto clasificado | 8 gestos predefinidos | Rápida |
| 05 | Pose Landmarks | 33 landmarks en 3D | 33 por persona | Rápida (lite) a lenta (heavy) |
| 06 | Holistic Landmarks | Pose + rostro + manos juntos | 553 en total | Más lenta |
| 07 | Object Detection | Cajas + categoría + score | Variable (depende del modelo) | Media |

Notas:
- "Velocidad" es aproximada, depende de la computadora, la resolución de captura y cuántos objetos haya en pantalla.
- Cuantos más landmarks devuelve un modelo, más lento suele ser.
- El Holistic Landmarker ejecuta varios modelos en cadena, por eso es el más lento de todos.

---

## 8. Convenciones de todos los scripts

Los scripts de este paquete comparten varias convenciones para que sean consistentes y fáciles de leer. Saber esto ayuda a moverse entre un script y otro sin perderse.

### 8.1. Estructura del archivo

Todos los scripts siguen el mismo orden interno:

1. **Docstring inicial**: describe qué hace el script, qué modelo usa,    qué devuelve, y los controles de teclado disponibles.
2. **Sección `1) PARÁMETROS EDITABLES DEL MODELO`**: todo lo que afecta cómo MediaPipe detecta (confianza mínima, resolución de captura, índice de cámara, cantidad máxima de objetos a detectar, etc.).
3. **Sección `2) PARÁMETROS EDITABLES DEL PANEL Y LA VISUALIZACIÓN`**: todo lo relacionado a qué se dibuja y cómo se ve el panel (colores, estados iniciales de los toggles, anchos, transparencia, etc.).
4. **Clase `ContadorFPS`**: calcula los FPS suavizados que se muestran en el panel de información.
5. **Funciones de dibujo**: dibujan los landmarks, cajas o contornos sobre el frame.
6. **Funciones de panel**: dibujan el panel de información (izquierda) y el panel de detalle en pausa (derecha).
7. **Función `procesar_tecla`**: interpreta las teclas presionadas y actualiza los toggles del diccionario `estado`.
8. **Función `main`**: el loop principal que captura frames, ejecuta el modelo, dibuja los resultados y maneja el teclado.

### 8.2. Los dos bloques de parámetros editables

La separación entre "parámetros del modelo" y "parámetros del panel" es deliberada: ayuda a distinguir qué cosas cambian el comportamiento del modelo (y por lo tanto los resultados) de qué cosas solo cambian cómo se ven en pantalla.

**Parámetros del modelo** (sección 1):

- `MODEL_FILENAME` / `MODEL_PATH`: el archivo del modelo.
- `MIN_*_CONFIDENCE`: los umbrales de confianza del modelo.
- `CAMARA_INDEX`: qué cámara usar.
- `ANCHO_CAPTURA` / `ALTO_CAPTURA`: la resolución pedida a la cámara.
- `NUM_*`: cuántos objetos detectar como máximo.

**Parámetros del panel** (sección 2):

- `MOSTRAR_*_INICIAL` / `DIBUJAR_*_INICIAL`: el estado inicial de cada toggle al arrancar el programa.
- `INTERVALO_ACTUALIZACION_FPS`: cada cuántos segundos se recalcula el número de FPS.
- `COLOR_*`: todos los colores en formato BGR.
- `ANCHO_PANEL*` / `ALPHA_PANEL`: el tamaño y la transparencia de los paneles.
- Constantes específicas de cada modelo (por ejemplo,
  `MAX_LANDMARKS_ROSTRO_PAUSA`).

### 8.3. Toggles comunes

Todos los scripts comparten al menos estos toggles:

| Tecla | Acción |
|-------|--------|
| `q` / `ESC` | Cerrar el programa |
| `i` | Mostrar / ocultar el panel de información |
| `ESPACIO` | Pausar / reanudar la captura |

Todos los toggles están documentados en el docstring del script y en su README específico.

### 8.4. La pausa

La pausa es una funcionalidad común a todos los scripts. Se activa con `ESPACIO`.

1. El script congela el último frame de la cámara.
2. Sigue analizándolo una y otra vez, pero el frame ya no cambia.
3. Aparece un panel de detalle en la esquina superior derecha con toda
   la información disponible del modelo: coordenadas normalizadas,
   confidencias, índices, y en algunos casos coordenadas adicionales
   como `visibility` o `presence`.
4. Los controles de toggle siguen funcionando: podés apagar y prender
   capas mientras está pausado para estudiar cada una por separado.

### 8.5. El panel de información

En la esquina superior izquierda de todos los scripts hay un panel
semitransparente con fondo oscuro que se dibuja sobre el video. Muestra:

- **FPS suavizados**: el número de frames por segundo, promediado
  durante medio segundo para que no salte tanto.
- **Estado ON/OFF de cada toggle**: cada capa visual tiene su fila, con
  el color verde si está activa y gris si está apagada.
- **Estado de pausa**: indica si estás en vivo o en pausa.
- **Resumen de la detección** (varía según el script): por ejemplo, la
  cantidad de manos detectadas y su lateralidad, o la confianza de cada
  rostro.

Se puede ocultar con la tecla `i`.

### 8.6. El panel de detalle (derecha)

En la esquina superior derecha, cuando corresponde, aparece un segundo
panel con información más detallada. Cuándo aparece depende del script.

### 8.7. Los timestamps

MediaPipe en modo VIDEO exige que cada frame tenga un timestamp
estrictamente mayor que el anterior, en milisegundos. Todos los scripts
resuelven esto con un **contador propio** que arranca en 0 y se
incrementa en 1 por cada frame:

```python
timestamp_ms += 1
```
Esto garantiza que nunca haya dos timestamps iguales, incluso cuando el
loop corre muy rápido (por ejemplo, en pausa, cuando no hay que esperar
a la cámara).

### 8.8. Cómo se dibujan los landmarks

Todos los scripts usan las utilidades de dibujo de mediapipe.tasks:
```python

mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing.draw_landmarks(
    image=frame,
    landmark_list=landmarks,
    connections=...,
    landmark_drawing_spec=...,
    connection_drawing_spec=...,
)
```

- `landmark_list`: la lista de puntos a dibujar.
- `connections`: qué puntos se conectan entre sí.
- `landmark_drawing_spec`: cómo se ve cada punto (color, grosor). Se puede pasar `None` para no dibujar los puntos.
- `connection_drawing_spec`: cómo se ven las líneas entre puntos.

Cada modelo tiene su propio conjunto de conexiones predefinidas. 

### 8.9. Orden de las operaciones en el loop

El loop principal de todos los scripts sigue este orden:
1. Capturar un frame (o usar el congelado si está pausado).
2. Espejar horizontalmente el frame (para que se sienta como un espejo).
3. Convertir BGR → RGB (OpenCV usa BGR, MediaPipe espera RGB).
4. Convertir el array de NumPy a un objeto mp.Image.
5. Incrementar el timestamp.
6. Ejecutar el modelo (detect_for_video, recognize_for_video, etc.).
7. Dibujar los landmarks sobre el frame.
8. Calcular los FPS.
9. Dibujar el panel de información.
10. Dibujar el panel de detalle (si corresponde).s
11. Mostrar el frame en una ventana.
12. Leer teclado y procesar la tecla.
13. Repetir.
