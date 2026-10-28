/*
===============================================================================
 RECEPTOR OSC + PARTÍCULAS (Processing)
===============================================================================

Versión en Processing de "receptor_particulas.py". Recibe los mensajes OSC
que manda "03_puntos_mano_osc.py" y los convierte en partículas.
La estructura es la misma que en la versión de Python, para poder
compararlas lado a lado

REQUISITOS
----------
  - Processing 4 (modo Java).
  - Librería oscP5:  Sketch > Importar librería > Gestionar librerías...
    buscar "oscP5" e instalar.

CÓMO PROBARLO
-------------
  1) Ejecutar este sketch (botón "play")
  2) Ejecutar 03_puntos_mano_osc.py
  3) Mover las manos frente a la cámara

QUÉ HACE CADA DATO (se puede cambiar en la función procesarManos())
-------------------------------------------------------------------
  palma        -> de dónde salen las partículas
  apertura     -> cuántas partículas salen y a qué velocidad
  inclinacion  -> color (girá la mano para recorrer el arco iris)
  cercania     -> tamaño de las partículas
  pinza        -> atrae las partículas de vuelta hacia la palma
  landmarks    -> se dibuja el esqueleto de la mano (tecla [l])

TECLAS
------
  l -> mostrar / ocultar el esqueleto de la mano
  i -> mostrar / ocultar el texto de información
  q -> salir

CÓMO FUNCIONA LA RECEPCIÓN OSC
------------------------------
  La librería oscP5 abre el puerto y llama a oscEvent() cada vez que llega
  un mensaje. oscEvent() NO dibuja: solo guarda el último valor recibido en
  los objetos `izquierda` y `derecha` (de la clase Mano).
  Después draw() lee esos objetos, crea partículas y dibuja.
  (oscP5 usa un hilo propio por dentro, pero no hace falta ocuparse de eso.)
===============================================================================
*/

import oscP5.*;
import netP5.*;


// =============================================================================
// 1) PARÁMETROS EDITABLES
// =============================================================================

// --- Red ---
final int PUERTO_OSC = 9000;      // el mismo que OSC_PUERTO en el emisor

// --- Manos ---
// Si pasan más de estos milisegundos sin recibir datos de una mano,
// se considera que desapareció.
final int TIMEOUT_MANO = 500;

// --- Partículas ---
final int   MAX_PARTICULAS = 2500;
final int   EMISION_MAX = 10;          // partículas por cuadro con la mano abierta
final float VELOCIDAD_MIN = 1.0;       // velocidad inicial con la mano cerrada
final float VELOCIDAD_MAX = 7.0;       // velocidad inicial con la mano abierta
final float RADIO_MIN = 2;             // tamaño con la mano lejos
final float RADIO_MAX = 10;            // tamaño con la mano cerca
final int   VIDA_MIN = 40;             // duración de cada partícula (en cuadros)
final int   VIDA_MAX = 90;
final float FUERZA_ATRACCION = 0.9;    // qué tan fuerte atrae la pinza
final float ROZAMIENTO = 0.98;         // 1.0 = sin frenado; menor = frena más

// Estela: en cada cuadro se dibuja un rectángulo negro semitransparente
// encima de todo. Cerca de 1.0 = estela larga; cerca de 0.5 = casi sin estela.
final float ESTELA = 0.86;

// Conexiones entre los 21 landmarks para dibujar el esqueleto de la mano
final int[][] CONEXIONES_MANO = {
  {0, 1},  {1, 2},   {2, 3},   {3, 4},     // pulgar
  {0, 5},  {5, 6},   {6, 7},   {7, 8},     // índice
  {5, 9},  {9, 10},  {10, 11}, {11, 12},   // medio
  {9, 13}, {13, 14}, {14, 15}, {15, 16},   // anular
  {13, 17}, {17, 18}, {18, 19}, {19, 20},  // meñique
  {0, 17}                                  // borde de la palma
};


// =============================================================================
// 2) VARIABLES GLOBALES
// =============================================================================

OscP5 oscP5;

// Acá se guarda lo último que llegó de cada mano
Mano derecha   = new Mano("derecha");
Mano izquierda = new Mano("izquierda");

ArrayList<Particula> particulas = new ArrayList<Particula>();

boolean mostrarEsqueleto = true;
boolean mostrarInfo = true;


// =============================================================================
// 3) SETUP Y DRAW
// =============================================================================

void setup() {
  size(720, 480);
  frameRate(60);

  // HSB con todos los rangos de 0 a 1:  color(tono, saturación, brillo, opacidad)
  // Es lo más cómodo para recorrer el arco iris con un solo número
  colorMode(HSB, 1, 1, 1, 1);
  background(0);

  // Abre el puerto y empieza a escuchar
  oscP5 = new OscP5(this, PUERTO_OSC);
}

void draw() {
  // 1) Estela: oscurecemos un poco lo dibujado en el cuadro anterior
  noStroke();
  fill(0, 0, 0, 1 - ESTELA);
  rect(0, 0, width, height);

  // 2) Crear partículas nuevas según los datos recibidos
  HashMap<String, float[]> atractores = procesarManos();

  // 3) Mover y dibujar todas las partículas (y eliminar las que se apagaron).
  //    Se recorre de atrás para adelante para poder borrar sin saltear ninguna.
  for (int i = particulas.size() - 1; i >= 0; i--) {
    Particula p = particulas.get(i);
    p.actualizar(atractores.get(p.dueno));
    p.dibujar();
    if (!p.viva()) {
      particulas.remove(i);
    }
  }

  // 4) Capas opcionales encima
  if (mostrarEsqueleto) {
    dibujarEsqueleto(izquierda);
    dibujarEsqueleto(derecha);
  }
  if (mostrarInfo) {
    dibujarInfo();
  }
}


// =============================================================================
// 4) LÓGICA: DE LOS DATOS OSC A LAS PARTÍCULAS
// =============================================================================
// ESTA ES LA FUNCIÓN PARA EXPERIMENTAR: acá se decide qué dato controla qué

HashMap<String, float[]> procesarManos() {
  // Un "atractor" por mano que está haciendo pinza: {x, y, fuerza}
  HashMap<String, float[]> atractores = new HashMap<String, float[]>();

  Mano[] todas = { izquierda, derecha };

  for (Mano m : todas) {

    // Ignorar manos que dejaron de enviar datos
    if (!m.activa()) continue;

    // --- Leer los datos (valores de 0 a 1, salvo inclinación) ---------------
    float x = m.palmaX * width;       // de 0-1 a píxeles
    float y = m.palmaY * height;

    // --- Traducir datos a propiedades visuales ------------------------------
    // apertura -> cantidad y velocidad
    int   cantidad  = int(1 + m.apertura * EMISION_MAX);
    float velocidad = mapear(m.apertura, 0, 1, VELOCIDAD_MIN, VELOCIDAD_MAX);

    // cercania -> tamaño
    float radio = mapear(m.cercania, 0, 1, RADIO_MIN, RADIO_MAX);

    // inclinacion -> color (de -90 a 90 grados recorre 80% del arco iris);
    // la mano derecha se desplaza medio arco iris para diferenciarse
    float tono = mapear(m.inclinacion, -90, 90, 0.0, 0.8);
    if (m.nombre.equals("derecha")) {
      tono += 0.5;
    }

    // pinza -> fuerza con la que se atraen las partículas de esta mano
    if (m.pinza > 0.3) {
      atractores.put(m.nombre, new float[] { x, y, m.pinza * FUERZA_ATRACCION });
    }

    // --- Crear las partículas nuevas ----------------------------------------
    for (int k = 0; k < cantidad; k++) {
      if (particulas.size() >= MAX_PARTICULAS) break;
      particulas.add(new Particula(x, y, velocidad, tono, radio, m.nombre));
    }
  }

  return atractores;
}


// =============================================================================
// 5) UTILIDAD
// =============================================================================

// Igual que map(), pero recorta el resultado para que no se salga del rango.
float mapear(float valor, float entradaMin, float entradaMax,
             float salidaMin, float salidaMax) {
  float t = constrain((valor - entradaMin) / (entradaMax - entradaMin), 0, 1);
  return salidaMin + t * (salidaMax - salidaMin);
}


// =============================================================================
// 6) DIBUJO DEL ESQUELETO Y DEL TEXTO
// =============================================================================

// Dibuja los 21 puntos de una mano usando el dato 'landmarks'
void dibujarEsqueleto(Mano m) {
  if (!m.activa() || !m.tieneLandmarks) return;

  // Los 63 valores vienen como x0 y0 z0 x1 y1 z1 ...
  // Para el punto i, x está en [i*3] e y en [i*3 + 1]. Usamos x e y.
  stroke(0, 0, 0.78);
  strokeWeight(1);
  for (int[] c : CONEXIONES_MANO) {
    line(m.landmarks[c[0] * 3] * width, m.landmarks[c[0] * 3 + 1] * height,
         m.landmarks[c[1] * 3] * width, m.landmarks[c[1] * 3 + 1] * height);
  }

  noStroke();
  fill(0, 0, 0.78);
  for (int i = 0; i < 21; i++) {
    ellipse(m.landmarks[i * 3] * width, m.landmarks[i * 3 + 1] * height, 6, 6);
  }
}

// Escribe en pantalla los valores que se están recibiendo
void dibujarInfo() {
  fill(0, 0, 0.9);
  textSize(13);
  textAlign(LEFT, TOP);

  text("Escuchando OSC en el puerto " + PUERTO_OSC, 15, 12);

  int linea = 1;
  Mano[] todas = { izquierda, derecha };
  for (Mano m : todas) {
    if (!m.activa()) continue;
    text(m.nombre
       + "   apertura " + nf(m.apertura, 1, 2)
       + "  pinza " + nf(m.pinza, 1, 2)
       + "  incl " + nf(m.inclinacion, 1, 1)
       + "  cerca " + nf(m.cercania, 1, 2)
       + "  dedos " + m.cantidadDedos, 15, 12 + linea * 20);
    linea++;
  }

  if (linea == 1) {
    text("Esperando manos... (ejecutar 03_puntos_mano_osc.py)", 15, 32);
  }
}

void keyPressed() {
  if (key == 'l' || key == 'L') mostrarEsqueleto = !mostrarEsqueleto;
  if (key == 'i' || key == 'I') mostrarInfo = !mostrarInfo;
  if (key == 'q' || key == 'Q') exit();
}


// =============================================================================
// 7) PARTÍCULAS
// =============================================================================

class Particula {
  float x, y;          // posición
  float vx, vy;        // velocidad
  float tono;          // color (0 a 1)
  float radio;
  String dueno;        // "derecha" o "izquierda"
  int vidaMax, vida;

  // Una partícula nace en un punto y sale en una dirección al azar
  Particula(float x, float y, float velocidad, float tono, float radio, String dueno) {
    float angulo = random(TWO_PI);
    float rapidez = velocidad * random(0.3, 1.0);

    this.x = x;
    this.y = y;
    this.vx = cos(angulo) * rapidez;
    this.vy = sin(angulo) * rapidez;
    this.tono = tono % 1.0;
    this.radio = radio;
    this.dueno = dueno;
    this.vidaMax = int(random(VIDA_MIN, VIDA_MAX + 1));
    this.vida = vidaMax;
  }

  // Mueve la partícula un paso.
  // `atractor` es null o {x, y, fuerza}: hacia dónde se la empuja.
  void actualizar(float[] atractor) {
    if (atractor != null) {
      float dx = atractor[0] - x;
      float dy = atractor[1] - y;
      float distancia = max(sqrt(dx * dx + dy * dy), 1.0);
      vx += dx / distancia * atractor[2];
      vy += dy / distancia * atractor[2];
    }

    x += vx;
    y += vy;
    vx *= ROZAMIENTO;
    vy *= ROZAMIENTO;
    vida--;
  }

  boolean viva() {
    return vida > 0;
  }

  void dibujar() {
    // A medida que pierde vida, se vuelve más oscura y más chica
    float factor = vida / (float) vidaMax;
    float r = max(1, int(radio * factor));

    noStroke();
    fill(tono, 0.85, factor);
    ellipse(x, y, r * 2, r * 2);
  }
}


// =============================================================================
// 8) RECEPCIÓN OSC
// =============================================================================
// oscEvent() se llama automáticamente cada vez que llega un mensaje.

void oscEvent(OscMessage mensaje) {
  // La dirección llega como texto: "/mano/derecha/apertura".
  // split() la corta en partes. Como empieza con "/", la primera parte
  // queda vacía:  ["", "mano", "derecha", "apertura"]
  String[] partes = split(mensaje.addrPattern(), '/');

  // Solo nos interesan los mensajes con forma /mano/<lado>/<dato>
  if (partes.length < 4 || !partes[1].equals("mano")) return;

  String lado = partes[2];
  String dato = partes[3];

  try {
    if (lado.equals("derecha")) {
      derecha.recibir(dato, mensaje);
    } else if (lado.equals("izquierda")) {
      izquierda.recibir(dato, mensaje);
    }
  } catch (Exception e) {
    println("Mensaje OSC inesperado: " + mensaje.addrPattern() + " -> " + e);
  }
}


// =============================================================================
// 9) CLASE MANO
// =============================================================================
// Guarda los últimos valores recibidos para una mano
// (en Python es el diccionario `manos`).

class Mano {
  String nombre;

  float palmaX = 0.5, palmaY = 0.5;
  float apertura = 0;
  float pinza = 0;
  float inclinacion = 0;
  float cercania = 0.5;
  int cantidadDedos = 0;

  float[] landmarks = new float[63];   // x0 y0 z0 x1 y1 z1 ...
  boolean tieneLandmarks = false;

  int ultimoMensaje = -100000;         // cuándo llegó el último mensaje (millis)

  Mano(String nombre) {
    this.nombre = nombre;
  }

  // ¿Llegaron datos hace poco?
  boolean activa() {
    return millis() - ultimoMensaje < TIMEOUT_MANO;
  }

  // Guarda el valor del mensaje según el nombre del dato.
  // floatValue() para decimales, intValue() para enteros.
  void recibir(String dato, OscMessage m) {
    ultimoMensaje = millis();

    if (dato.equals("palma")) {
      palmaX = m.get(0).floatValue();
      palmaY = m.get(1).floatValue();
    } else if (dato.equals("apertura")) {
      apertura = m.get(0).floatValue();
    } else if (dato.equals("pinza")) {
      pinza = m.get(0).floatValue();
    } else if (dato.equals("inclinacion")) {
      inclinacion = m.get(0).floatValue();
    } else if (dato.equals("cercania")) {
      cercania = m.get(0).floatValue();
    } else if (dato.equals("cantidad_dedos")) {
      cantidadDedos = m.get(0).intValue();
    } else if (dato.equals("landmarks")) {
      for (int i = 0; i < 63; i++) {
        landmarks[i] = m.get(i).floatValue();
      }
      tieneLandmarks = true;
    }
  }
}
