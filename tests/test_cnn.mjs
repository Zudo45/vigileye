// Verifica que el modelo en TensorFlow.js reproduce las salidas calculadas en Python (models/referencia.json).
// Requiere:  npm install  (instala @tensorflow/tfjs solo para esta prueba)   y luego   node tests/test_cnn.mjs
import fs from 'node:fs';
import * as tf from '@tensorflow/tfjs';
import { buildModel, selfTest } from '../js/cnn.js';

const dir = new URL('../models/', import.meta.url);
const meta = JSON.parse(fs.readFileSync(new URL('modelo_meta.json', dir), 'utf8'));
const ref = JSON.parse(fs.readFileSync(new URL('referencia.json', dir), 'utf8'));
const buf = fs.readFileSync(new URL('pesos.bin', dir));
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

const model = buildModel(tf, ab, meta);
console.log('Parámetros del modelo:', model.countParams());
const r = selfTest(tf, model, ref, 1e-3);
console.log('Salida JS    :', r.salida.map(v => v.toFixed(5)).join('  '));
console.log('Salida Python:', r.esperado.map(v => v.toFixed(5)).join('  '));
console.log('Diferencia máxima:', r.maxDiff.toExponential(2));
if (!r.ok) { console.error('FALLÓ: el modelo en JavaScript no coincide con el de Python'); process.exit(1); }
console.log('OK   el modelo en TensorFlow.js coincide con Python');
