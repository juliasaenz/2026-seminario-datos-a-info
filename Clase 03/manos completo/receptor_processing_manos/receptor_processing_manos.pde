/*
===============================================================================
 RECEPTOR OSC SIMPLE (Processing): PUNTOS DE LA MANO + PANEL DE DATOS
===============================================================================

Recibe los mensajes OSC que manda "03_puntos_mano_osc.py", vuelve a dibujar
los 21 puntos de cada mano y muestra TODOS los datos recibidos en un panel
a la derecha. Sirve para ver qué llega por OSC antes de usarlo en otro
proyecto.

REQUISITOS
----------
  - Processing 4 (modo Java)
  - Librería oscP5:  Sketch > Importar librería > Gestionar librerías...
    buscar "oscP5" e instalar

CÓMO PROBARLO
-------------
  1) Ejecutar este sketch (botón "play")
  2) Ejecutar 03_puntos_mano_osc.py
  3) Mostrar las manos a la cámara

QUÉ SE VE
---------
  Izquierda: los 21 puntos de cada mano conectados (mano izquierda en un
             color, derecha en otro) y un circulito blanco en la palma
  Derecha:   un panel por mano con
               - confianza de la lateralidad
               - palma (x, y)
               - apertura, pinza y cercanía (barras de 0 a 1)
               - inclinación (en grados)
               - qué dedos están extendidos: P=pulgar, I=índice, M=medio,
                 A=anular, m=meñique
               - tabla con los 21 landmarks crudos (x, y, z)

TECLAS
------
  n -> mostrar / ocultar el número de cada punto
===============================================================================
*/

import oscP5.*;
import netP5.*;


// =============================================================================
// 1) PARÁMETROS EDITABLES
// =============================================================================

final int PUERTO_OSC = 9000;     // el mismo que OSC_PUERTO en el emisor

// Si pasan más de estos milisegundos sin datos de una mano,
// se considera que desapareció
final int TIMEOUT_MANO = 500;

// El panel ocupa este ancho a la derecha. El resto de la ventana
// (ancho total - ANCHO_PANEL) es el área donde se dibujan las manos
final int ANCHO_PANEL = 380;

// Conexiones entre los 21 landmarks para dibujar el esqueleto de la mano
final int[][] CONEXIONES_MANO = {
  {0, 1},  {1, 2},   {2, 3},   {3, 4},     // pulgar
  {0, 5},  {5, 6},   {6, 7},   {7, 8},     // índice
  {5, 9},  {9, 10},  {10, 11}, {11, 12},   // medio
  {9, 13}, {13, 14}, {14, 15}, {15, 16},   // anular
  {13, 17}, {17, 18}, {18, 19}, {19, 20},  // meñique
  {0, 17}                                  // borde de la palma
};

// Letras para los 5 dedos, en el orden en que los manda el emisor
final String[] LETRAS_DEDOS = { "P", "I", "M", "A", "m" };


// =============================================================================
// 2) VARIABLES GLOBALES
// =============================================================================

OscP5 oscP5;

// Acá se guarda lo último que llegó de cada mano
Mano izquierda = new Mano("izquierda");
Mano derecha   = new Mano("derecha");

int cantidadManos = 0;        // dato /mano/cantidad
boolean mostrarNumeros = false;

color colorIzquierda, colorDerecha;
PFont fuenteNormal, fuenteMono;


// =============================================================================
// 3) SETUP Y DRAW
// =============================================================================

void setup() {
  size(900, 520);

  colorIzquierda = color(80, 200, 255);    // celeste
  colorDerecha   = color(255, 120, 200);   // rosa

  fuenteNormal = createFont("SansSerif", 12);
  fuenteMono   = createFont("Monospaced", 10);

  // Abre el puerto y empieza a escuchar
  oscP5 = new OscP5(this, PUERTO_OSC);
}

void draw() {
  background(18);
  textAlign(LEFT, TOP);

  dibujarAreaManos();
  dibujarPanel();
}

void keyPressed() {
  if (key == 'n' || key == 'N') {
    mostrarNumeros = !mostrarNumeros;
  }
}


// =============================================================================
// 4) ÁREA DE MANOS
// =============================================================================

void dibujarAreaManos() {
  float anchoArea = width - ANCHO_PANEL;

  // Marco del área
  noFill();
  stroke(60);
  strokeWeight(1);
  rect(0, 0, anchoArea - 1, height - 1);

  // Mensaje de espera
  if (!izquierda.activa() && !derecha.activa()) {
    fill(120);
    noStroke();
    textFont(fuenteNormal);
    textSize(16);
    textAlign(CENTER, CENTER);
    text("Esperando manos...\n(ejecutar 03_puntos_mano_osc.py)", anchoArea / 2, height / 2);
    textAlign(LEFT, TOP);
  }

  dibujarMano(izquierda, colorIzquierda);
  dibujarMano(derecha, colorDerecha);
}

// Dibuja los 21 puntos de una mano y las líneas que los conectan
// Los datos llegan normalizados (0 a 1): multiplicamos por el tamaño
// del área para pasarlos a píxeles
void dibujarMano(Mano m, color c) {
  if (!m.activa() || !m.tieneLandmarks) return;

  float anchoArea = width - ANCHO_PANEL;

  // Líneas
  stroke(c);
  strokeWeight(2);
  for (int[] con : CONEXIONES_MANO) {
    line(m.puntoX(con[0]) * anchoArea, m.puntoY(con[0]) * height,
         m.puntoX(con[1]) * anchoArea, m.puntoY(con[1]) * height);
  }

  // Puntos
  noStroke();
  fill(c);
  for (int i = 0; i < 21; i++) {
    ellipse(m.puntoX(i) * anchoArea, m.puntoY(i) * height, 8, 8);
  }

  // Centro de la palma (dato derivado "palma")
  noFill();
  stroke(255);
  strokeWeight(1);
  ellipse(m.palmaX * anchoArea, m.palmaY * height, 16, 16);

  // Número de cada punto (tecla n)
  if (mostrarNumeros) {
    fill(255);
    noStroke();
    textFont(fuenteNormal);
    textSize(10);
    for (int i = 0; i < 21; i++) {
      text(i, m.puntoX(i) * anchoArea + 6, m.puntoY(i) * height - 12);
    }
  }
}


// =============================================================================
// 5) PANEL DE DATOS
// =============================================================================

void dibujarPanel() {
  float x0 = width - ANCHO_PANEL;

  // Fondo del panel
  noStroke();
  fill(28);
  rect(x0, 0, ANCHO_PANEL, height);

  // Encabezado
  fill(200);
  textFont(fuenteNormal);
  textSize(12);
  text("OSC puerto " + PUERTO_OSC + "    manos detectadas: " + cantidadManos, x0 + 12, 12);

  // Una columna por mano
  dibujarPanelMano(izquierda, "IZQUIERDA", x0 + 12,  40, colorIzquierda);
  dibujarPanelMano(derecha,   "DERECHA",   x0 + 198, 40, colorDerecha);
}

// Panel con todos los datos de UNA mano, empezando en (x, y)
void dibujarPanelMano(Mano m, String titulo, float x, float y, color c) {
  textFont(fuenteNormal);
  textAlign(LEFT, TOP);

  // Título (gris si la mano no está)
  textSize(13);
  fill(m.activa() ? c : color(90));
  text(titulo, x, y);

  if (!m.activa()) {
    fill(90);
    textSize(12);
    text("sin datos", x, y + 20);
    return;
  }
  y += 22;

  // --- Datos sueltos -------------------------------------------------------
  textSize(12);
  fill(220);
  text("confianza " + nf(m.confianza, 1, 2), x, y);
  y += 16;
  text("palma " + nf(m.palmaX, 1, 2) + "  " + nf(m.palmaY, 1, 2), x, y);
  y += 18;

  // --- Barras (valores de 0 a 1) ---------------------------------------------
  dibujarBarra("apertura", m.apertura, x, y, 170, c);
  y += 26;
  dibujarBarra("pinza", m.pinza, x, y, 170, c);
  y += 26;
  dibujarBarra("cercania", m.cercania, x, y, 170, c);
  y += 26;

  // --- Inclinación -----------------------------------------------------------
  fill(220);
  text("inclinacion " + nf(m.inclinacion, 1, 1) + " grados", x, y);
  y += 20;

  // --- Dedos extendidos: cuadradito lleno = extendido ----------------------
  for (int i = 0; i < 5; i++) {
    float bx = x + i * 24;
    stroke(c);
    strokeWeight(1);
    if (m.dedos[i] == 1) fill(c); else noFill();
    rect(bx, y, 18, 18);

    // Letra dentro del cuadradito
    noStroke();
    fill(m.dedos[i] == 1 ? color(20) : c);
    textAlign(CENTER, CENTER);
    text(LETRAS_DEDOS[i], bx + 9, y + 9);
    textAlign(LEFT, TOP);
  }
  y += 24;
  fill(220);
  text("dedos extendidos: " + m.cantidadDedos, x, y);
  y += 24;

  // --- Tabla de landmarks crudos -------------------------------------------
  if (!m.tieneLandmarks) return;

  textFont(fuenteMono);
  textSize(10);
  fill(150);
  text(" #     x     y     z", x, y);
  y += 14;

  fill(200);
  for (int i = 0; i < 21; i++) {
    // nfs() deja un espacio delante de los positivos para que alineen
    text(nf(i, 2) + " " + nfs(m.puntoX(i), 1, 2) + " "
                        + nfs(m.puntoY(i), 1, 2) + " "
                        + nfs(m.puntoZ(i), 1, 2), x, y);
    y += 12;
  }
}

// Una barra horizontal para un valor de 0 a 1
void dibujarBarra(String nombre, float valor, float x, float y, float ancho, color c) {
  textFont(fuenteNormal);
  textSize(12);
  fill(220);
  noStroke();
  text(nombre + " " + nf(valor, 1, 2), x, y);

  // Contorno
  noFill();
  stroke(90);
  strokeWeight(1);
  rect(x, y + 16, ancho, 7);

  // Relleno
  noStroke();
  fill(c);
  rect(x, y + 16, ancho * constrain(valor, 0, 1), 7);
}


// =============================================================================
// 6) RECEPCIÓN OSC
// =============================================================================
// oscEvent() se llama automáticamente cada vez que llega un mensaje.
// Solo guarda los valores; el dibujo se hace siempre en draw().

void oscEvent(OscMessage mensaje) {
  // La dirección llega como texto: "/mano/derecha/apertura".
  // split() la corta en partes. Como empieza con "/", la primera parte
  // queda vacía:  ["", "mano", "derecha", "apertura"]
  String[] partes = split(mensaje.addrPattern(), '/');

  if (partes.length < 3 || !partes[1].equals("mano")) return;

  try {
    // Mensaje /mano/cantidad  (solo tiene 3 partes)
    if (partes.length == 3) {
      if (partes[2].equals("cantidad")) {
        cantidadManos = mensaje.get(0).intValue();
      }
      return;
    }

    // Mensajes /mano/<lado>/<dato>
    String lado = partes[2];
    String dato = partes[3];

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
// 7) CLASE MANO
// =============================================================================
// Guarda los últimos valores recibidos para una mano.

class Mano {
  String nombre;

  // Datos derivados
  float confianza = 0;
  float palmaX = 0.5, palmaY = 0.5;
  float apertura = 0;
  float pinza = 0;
  float inclinacion = 0;
  float cercania = 0;
  int cantidadDedos = 0;
  int[] dedos = new int[5];          // pulgar, índice, medio, anular, meñique

  // Datos crudos: 21 puntos x 3 valores (x, y, z)
  float[] landmarks = new float[63];
  boolean tieneLandmarks = false;

  int ultimoMensaje = -100000;       // cuándo llegó el último mensaje (millis)

  Mano(String nombre) {
    this.nombre = nombre;
  }

  // ¿Llegaron datos hace poco?
  boolean activa() {
    return millis() - ultimoMensaje < TIMEOUT_MANO;
  }

  // Coordenadas del landmark número i (los datos están como x0 y0 z0 x1 y1 z1 ...)
  float puntoX(int i) { return landmarks[i * 3]; }
  float puntoY(int i) { return landmarks[i * 3 + 1]; }
  float puntoZ(int i) { return landmarks[i * 3 + 2]; }

  // Guarda el valor del mensaje según el nombre del dato.
  // floatValue() para decimales, intValue() para enteros.
  void recibir(String dato, OscMessage m) {
    ultimoMensaje = millis();

    if (dato.equals("confianza")) {
      confianza = m.get(0).floatValue();
    } else if (dato.equals("palma")) {
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
    } else if (dato.equals("dedos")) {
      for (int i = 0; i < 5; i++) {
        dedos[i] = m.get(i).intValue();
      }
    } else if (dato.equals("landmarks")) {
      for (int i = 0; i < 63; i++) {
        landmarks[i] = m.get(i).floatValue();
      }
      tieneLandmarks = true;
    }
  }
}
