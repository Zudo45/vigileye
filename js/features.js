/**
 * features.js - Medidas faciales a partir de los landmarks de MediaPipe (puerto de features.py, con postura ampliada).
 * EAR, MAR, ángulos de la cabeza (pitch/yaw/roll), proporción alto/ancho del rostro,
 * inclinación lateral por la línea de los ojos y recorte de ojo de 64x64 para la CNN.
 */

// Índices de MediaPipe Face Mesh (478 puntos)
// ojo: p1 (esquina), p2, p3 (párpado sup.), p4 (esquina), p5, p6 (párpado inf.)
export const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
export const LEFT_EYE = [362, 385, 387, 263, 373, 380];
// boca (labio interno): esquina izq., sup-izq, sup-centro, sup-der, esquina der., inf-der, inf-centro, inf-izq
export const MOUTH = [78, 81, 13, 311, 308, 402, 14, 178];
// postura: frente (10), mentón (152), mejillas junto a las orejas (234 der., 454 izq.), nariz (1),
// esquinas externas de los ojos (33 der., 263 izq.)
export const FOREHEAD = 10, CHIN = 152, CHEEK_R = 234, CHEEK_L = 454, NOSE = 1, EYE_OUT_R = 33, EYE_OUT_L = 263;

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** EAR de Soukupová y Čech (2016). pts: 6 puntos [x, y] en píxeles. */
export function eyeAspectRatio(pts) {
  const [p1, p2, p3, p4, p5, p6] = pts;
  return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4));
}

/** MAR: promedio de 3 aperturas verticales / ancho de la boca. */
export function mouthAspectRatio(pts) {
  const [left, ul, um, ur, right, lr, lm, ll] = pts;
  return (dist(ul, ll) + dist(um, lm) + dist(ur, lr)) / (3.0 * dist(left, right));
}

/**
 * Ángulos (pitch, yaw, roll) en grados a partir de la matriz 4x4 de MediaPipe.
 * `data` llega en orden de columnas (column-major), como entrega MediaPipe en JavaScript.
 * El signo depende de la convención de ejes; el sistema usa la DESVIACIÓN respecto a la
 * postura de calibración, no el valor absoluto.
 */
export function eulerFromMatrix(data) {
  const R = [[data[0], data[4], data[8]], [data[1], data[5], data[9]], [data[2], data[6], data[10]]];
  for (let j = 0; j < 3; j++) {            // quita la escala de cada columna
    const n = Math.hypot(R[0][j], R[1][j], R[2][j]) || 1;
    for (let i = 0; i < 3; i++) R[i][j] /= n;
  }
  const sy = Math.hypot(R[0][0], R[1][0]);
  const deg = 180 / Math.PI;
  return {
    pitch: Math.atan2(R[2][1], R[2][2]) * deg,
    yaw: Math.atan2(-R[2][0], sy) * deg,
    roll: Math.atan2(R[1][0], R[0][0]) * deg,
  };
}

/** Diferencia a-b en grados, envuelta a [-180, 180]. */
export const angleDiff = (a, b) => ((a - b + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;

/**
 * Recorte cuadrado centrado en el ojo -> Float32Array de size*size con niveles de gris 0-255
 * (misma entrada que el clasificador). Devuelve null si el recorte es demasiado pequeño.
 * `source` es un <video>; `ctx` un contexto 2D de un canvas de size x size reutilizable.
 */
export function eyeCrop(source, ctx, pts, size = 64) {
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const half = Math.max(Math.max(...xs) - Math.min(...xs), 1.0) * 0.9;     // margen alrededor del ojo
  const w = source.videoWidth, h = source.videoHeight;
  const x0 = Math.trunc(Math.max(cx - half, 0)), x1 = Math.trunc(Math.min(cx + half, w));
  const y0 = Math.trunc(Math.max(cy - half, 0)), y1 = Math.trunc(Math.min(cy + half, h));
  if (x1 - x0 < 4 || y1 - y0 < 4) return null;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, x0, y0, x1 - x0, y1 - y0, 0, 0, size, size);
  const img = ctx.getImageData(0, 0, size, size).data;
  const out = new Float32Array(size * size);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    out[i] = 0.299 * img[j] + 0.587 * img[j + 1] + 0.114 * img[j + 2];       // mismo peso que OpenCV BGR2GRAY
  }
  return out;
}

/**
 * Caja del recorte de ojo, normalizada [y1, x1, y2, x2] (para tf.image.cropAndResize), con la misma geometría de eyeCrop.
 * Devuelve null si el recorte es demasiado pequeño.
 */
export function eyeBox(pts, w, h) {
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const half = Math.max(Math.max(...xs) - Math.min(...xs), 1.0) * 0.9;
  const x0 = Math.trunc(Math.max(cx - half, 0)), x1 = Math.trunc(Math.min(cx + half, w));
  const y0 = Math.trunc(Math.max(cy - half, 0)), y1 = Math.trunc(Math.min(cy + half, h));
  if (x1 - x0 < 4 || y1 - y0 < 4) return null;
  return [y0 / h, x0 / w, y1 / h, x1 / w];
}

/**
 * A partir del resultado de FaceLandmarker devuelve las medidas del cuadro o null si no hay rostro.
 * Puntos en píxeles del video. `hw` = alto/ancho del rostro: baja al cabecear, no cambia al alejarse
 * y sube un poco al girar. `rollEye` = ángulo de la línea entre las esquinas externas de los ojos.
 */
export function extractFeatures(result, width, height) {
  if (!result || !result.faceLandmarks || !result.faceLandmarks.length) return null;
  const lm = result.faceLandmarks[0];
  const P = idx => idx.map(i => [lm[i].x * width, lm[i].y * height]);
  const pt = i => [lm[i].x * width, lm[i].y * height];
  const right = P(RIGHT_EYE), left = P(LEFT_EYE), mouth = P(MOUTH);
  const forehead = pt(FOREHEAD), chin = pt(CHIN), cheekR = pt(CHEEK_R), cheekL = pt(CHEEK_L), nose = pt(NOSE);
  const eyeR = pt(EYE_OUT_R), eyeL = pt(EYE_OUT_L);
  const out = {
    ear: (eyeAspectRatio(right) + eyeAspectRatio(left)) / 2.0,
    mar: mouthAspectRatio(mouth),
    hw: dist(forehead, chin) / Math.max(dist(cheekR, cheekL), 1e-6),
    rollEye: Math.atan2(eyeL[1] - eyeR[1], eyeL[0] - eyeR[0]) * 180 / Math.PI,
    pitch: null, yaw: null, roll: null,
    rightEye: right, leftEye: left, mouth,
    posture: { forehead, chin, cheekR, cheekL, nose },
  };
  const m = result.facialTransformationMatrixes;
  if (m && m.length && m[0].data) Object.assign(out, eulerFromMatrix(m[0].data));
  return out;
}
