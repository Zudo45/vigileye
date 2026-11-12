// Escenarios simulados del motor de reglas (equivalen a test_rules.py + postura de la cabeza).
// Ejecutar:  node tests/test_rules.mjs
import assert from 'node:assert/strict';
import { RuleEngine, eyeClosedScore, decideEyeClosed, DEFAULT_CONFIG, NORMAL, ATENCION, RIESGO, ALERTA, SIN_ROSTRO } from '../js/rules.js';

const FPS = 30;
function run(seconds, sig = {}) {
  const eng = new RuleEngine();
  const levels = [];
  for (let i = 0; i < seconds * FPS; i++) {
    const t = i / FPS;
    const args = {};
    for (const [k, v] of Object.entries(sig)) args[k] = typeof v === 'function' ? v(t) : v;
    levels.push([t, eng.update(t, true, args)]);
  }
  return { eng, levels };
}
const firstTime = (levels, lvl) => { const f = levels.find(([, l]) => l === lvl); return f ? f[0] : null; };
const tests = {
  parpadeo_normal_no_alarma() {
    const { levels } = run(60, { eyeClosed: t => (t % 4.0) < 0.2 });
    assert.ok(levels.every(([, l]) => l === NORMAL));
  },
  microsueno_sin_fatiga_exige_4_segundos() {
    const { levels } = run(20, { eyeClosed: t => t >= 10.0 });
    const ta = firstTime(levels, ALERTA);
    assert.ok(ta !== null && Math.abs(ta - 14.0) < 0.1, `t_alerta=${ta}`);
  },
  cierre_de_3_segundos_sin_fatiga_no_alerta() {
    const { levels } = run(20, { eyeClosed: t => t >= 10.0 && t < 13.0 });   // p. ej. un movimiento brusco
    assert.equal(firstTime(levels, ALERTA), null);
  },
  microsueno_con_fatiga_acumulada_es_inmediato() {
    // dos bostezos y luego ojos cerrados desde t=12: con contexto de fatiga basta 2 s
    const { levels } = run(25, { marDelta: t => ((t >= 2 && t < 4) || (t >= 6 && t < 8) ? 0.6 : 0.0), eyeClosed: t => t >= 12.0 });
    const ta = firstTime(levels, ALERTA);
    assert.ok(ta !== null && Math.abs(ta - 14.0) < 0.1, `t_alerta=${ta}`);
  },
  hablar_no_es_bostezo() {
    const { eng, levels } = run(30, { marDelta: t => ((t % 2.0) < 0.5 ? 0.5 : 0.0) });
    assert.equal(eng.info.yawns, 0);
    assert.ok(levels.every(([, l]) => l === NORMAL));
  },
  un_solo_bostezo_no_es_senal() {
    const { levels } = run(20, { marDelta: t => (t >= 5 && t < 8 ? 0.6 : 0.0) });
    assert.ok(levels.every(([, l]) => l === NORMAL));
  },
  dos_bostezos_en_un_minuto_dan_atencion() {
    const { levels } = run(30, { marDelta: t => ((t >= 5 && t < 8) || (t >= 15 && t < 18) ? 0.6 : 0.0) });
    assert.ok(firstTime(levels, ATENCION) !== null);
    assert.equal(firstTime(levels, ALERTA), null);
  },
  bostezos_mas_cabeceo_es_riesgo_y_luego_alerta() {
    // dos bostezos y cabeceo sostenido desde t=8
    const { levels } = run(30, { marDelta: t => ((t >= 2 && t < 4) || (t >= 6 && t < 8) ? 0.6 : 0.0), pitchDev: t => (t >= 8 ? 25.0 : 0.0) });
    const tr = firstTime(levels, RIESGO), ta = firstTime(levels, ALERTA);
    assert.ok(tr !== null && ta !== null);
    assert.ok(ta - tr >= 4.9, `riesgo=${tr} alerta=${ta}`);
  },
  perclos_alto_da_atencion() {
    const { levels } = run(70, { eyeClosed: t => (t % 5.0) < 1.0 });
    assert.ok(firstTime(levels, ATENCION) !== null);
    assert.equal(firstTime(levels, ALERTA), null);
  },
  sin_rostro() {
    const eng = new RuleEngine();
    let lvl = NORMAL;
    for (let i = 0; i < 5 * FPS; i++) lvl = eng.update(i / FPS, false);
    assert.equal(lvl, SIN_ROSTRO);
  },
  fusion() {
    const c = DEFAULT_CONFIG;
    assert.ok(eyeClosedScore(1.0, 0.0, c) < 0.1);
    assert.ok(eyeClosedScore(0.45, 0.95, c) > 0.9);
    assert.equal(eyeClosedScore(0.45, null, c), 1.0);
  },
  cnn_filtra_falsos_cierres_por_movimiento() {
    // el EAR dice 'cerrado' (0,45) pero la CNN ve el ojo abierto (0,12): no es un cierre
    assert.equal(decideEyeClosed(0.45, 0.12).closed, false);
    // ambos coinciden: cierre
    assert.equal(decideEyeClosed(0.45, 0.97).closed, true);
    // ojo abierto en ambos
    assert.equal(decideEyeClosed(0.99, 0.00).closed, false);
    // la CNN sola no basta: el EAR también debe indicar cierre
    assert.equal(decideEyeClosed(0.95, 0.99).closed, false);
    // sin CNN se usa solo el EAR
    assert.equal(decideEyeClosed(0.45, null).closed, true);
  },
  // ---- postura de la cabeza
  cabeceo_por_proporcion_da_atencion() {
    const { levels } = run(15, { hwRel: t => (t >= 5 ? 0.7 : 1.0) });       // la cara se "recorta" en vertical
    assert.ok(firstTime(levels, ATENCION) !== null);
    assert.equal(firstTime(levels, ALERTA), null);
  },
  girar_al_retrovisor_no_alerta() {
    // giro breve (1.5 s) y proporción alto/ancho que sube: no debe cambiar el nivel
    const { levels } = run(20, { yawDev: t => (t >= 5 && t < 6.5 ? 45 : 0), hwRel: t => (t >= 5 && t < 6.5 ? 1.2 : 1.0) });
    assert.ok(levels.every(([, l]) => l === NORMAL));
  },
  mirada_lateral_sostenida_es_solo_atencion() {
    const { levels } = run(20, { yawDev: t => (t >= 5 ? 50 : 0) });
    assert.ok(firstTime(levels, ATENCION) !== null);
    assert.equal(firstTime(levels, RIESGO), null);
    assert.equal(firstTime(levels, ALERTA), null);
  },
  inclinacion_lateral_sostenida_da_atencion() {
    const { levels } = run(15, { rollDev: t => (t >= 5 ? 35 : 0) });
    assert.ok(firstTime(levels, ATENCION) !== null);
  },
  cabeza_caida_con_rostro_perdido_es_alerta() {
    const eng = new RuleEngine();
    let lvl = NORMAL, t = 0;
    for (; t < 5; t += 1 / FPS) lvl = eng.update(t, true, { pitchDev: t > 3 ? 20 : 0, hwRel: t > 3 ? 0.8 : 1.0 });
    for (let i = 0; i < 3 * FPS; i++, t += 1 / FPS) lvl = eng.update(t, false);
    assert.equal(lvl, ALERTA);
    assert.equal(eng.info.cabeza_caida, true);
  },
  rostro_perdido_sin_inclinacion_no_es_alerta() {
    const eng = new RuleEngine();
    let lvl = NORMAL, t = 0;
    for (; t < 5; t += 1 / FPS) lvl = eng.update(t, true);
    for (let i = 0; i < 4 * FPS; i++, t += 1 / FPS) lvl = eng.update(t, false);
    assert.equal(lvl, SIN_ROSTRO);
  },
};

let n = 0;
for (const [name, fn] of Object.entries(tests)) { fn(); console.log('OK  ', name); n++; }
console.log(`\n${n} escenarios superados`);
