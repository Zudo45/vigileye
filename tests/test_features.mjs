// Pruebas de las medidas geométricas. Ejecutar:  node tests/test_features.mjs
import assert from 'node:assert/strict';
import { eyeAspectRatio, mouthAspectRatio, eulerFromMatrix, angleDiff, extractFeatures, eyeBox } from '../js/features.js';

const close = (a, b, tol = 1e-6) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);

// EAR: ojo abierto (alto 2, ancho 4) -> (2+2)/(2*4) = 0.5 ; cerrado -> 0
const open = [[0, 0], [1, -1], [3, -1], [4, 0], [3, 1], [1, 1]];
const closed = [[0, 0], [1, 0], [3, 0], [4, 0], [3, 0], [1, 0]];
close(eyeAspectRatio(open), 0.5); close(eyeAspectRatio(closed), 0);
console.log('OK   EAR');

// MAR: boca con 3 aperturas de 3 y ancho de 6 -> 0.5
const mouth = [[0, 0], [1, -1.5], [3, -1.5], [5, -1.5], [6, 0], [5, 1.5], [3, 1.5], [1, 1.5]];
close(mouthAspectRatio(mouth), 0.5);
console.log('OK   MAR');

// Ángulos: matriz (column-major) de una rotación pura
const colMajor = R => [R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0, R[0][2], R[1][2], R[2][2], 0, 0, 0, 0, 1];
const rad = d => d * Math.PI / 180, c = Math.cos, s = Math.sin;
const Rx = a => [[1, 0, 0], [0, c(a), -s(a)], [0, s(a), c(a)]];
const Ry = a => [[c(a), 0, s(a)], [0, 1, 0], [-s(a), 0, c(a)]];
const Rz = a => [[c(a), -s(a), 0], [s(a), c(a), 0], [0, 0, 1]];
let e = eulerFromMatrix(colMajor(Rx(rad(30))));  close(e.pitch, 30, 1e-6); close(e.yaw, 0, 1e-6); close(e.roll, 0, 1e-6);
e = eulerFromMatrix(colMajor(Ry(rad(25))));      close(e.yaw, 25, 1e-6);
e = eulerFromMatrix(colMajor(Rz(rad(-15))));     close(e.roll, -15, 1e-6);
// con escala (la matriz de MediaPipe trae escala): el resultado no cambia
const scaled = colMajor(Rx(rad(30))).map((v, i) => (i % 4 === 3 || i > 11 ? v : v * 7.5));
close(eulerFromMatrix(scaled).pitch, 30, 1e-6);
console.log('OK   ángulos de la cabeza');

close(angleDiff(170, -170), -20); close(angleDiff(-170, 170), 20); close(angleDiff(10, 5), 5);
console.log('OK   diferencia de ángulos');

// extractFeatures con 478 puntos sintéticos: rostro de 200 de ancho y 260 de alto en un video de 640x480
const lm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
const set = (i, x, y) => { lm[i] = { x: x / 640, y: y / 480 }; };
set(10, 320, 110); set(152, 320, 370); set(234, 220, 240); set(454, 420, 240);
set(33, 270, 200); set(263, 370, 200);
const f = extractFeatures({ faceLandmarks: [lm], facialTransformationMatrixes: [{ data: colMajor(Rx(rad(10))) }] }, 640, 480);
close(f.hw, 260 / 200); close(f.rollEye, 0); close(f.pitch, 10, 1e-6);
// cabeza inclinada: ojos con 10 px de diferencia vertical sobre 100 de ancho
set(263, 370, 210);
close(extractFeatures({ faceLandmarks: [lm] }, 640, 480).rollEye, Math.atan2(10, 100) * 180 / Math.PI, 1e-6);
assert.equal(extractFeatures({ faceLandmarks: [] }, 640, 480), null);
console.log('OK   extractFeatures (proporción alto/ancho, inclinación lateral)');
console.log('\nPruebas de medidas superadas');
