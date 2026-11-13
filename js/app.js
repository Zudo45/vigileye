/**
 * app.js - VigilEye en el navegador (puerto de app.py).
 *
 * Flujo por cuadro:
 *   cámara -> MediaPipe FaceLandmarker -> EAR / MAR / pose / proporción del rostro / recortes de ojo
 *   -> calibración (10 s) -> suavizado -> CNN + EAR (fusión) -> motor de reglas
 *   -> nivel (NORMAL / ATENCION / RIESGO / ALERTA) -> alarma + registro CSV
 * Todo se procesa en este dispositivo; el video no se envía ni se guarda.
 */
import { RuleEngine, DEFAULT_CONFIG, decideEyeClosed, NORMAL, ATENCION, RIESGO, ALERTA, SIN_ROSTRO } from './rules.js?v=6';
import { extractFeatures, eyeBox, angleDiff, coverTransform } from './features.js?v=6';
import { buildModel, predictFromFrame, selfTest } from './cnn.js?v=6';

const MP_VERSION = '0.10.14';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const CALIB_S = 10.0;          // duración de la calibración inicial (s)
const MIN_CALIB_FRAMES = 30;
const SMOOTH_N = 5;            // ventana del promedio móvil (cuadros)

// Opciones de desarrollo (CNN sí/no y video grabado): se activan abriendo la página con ?dev=1
const DEV_PARAM = new URLSearchParams(location.search).get('dev');      // ?dev=1 diagnóstico y CNN; ?dev=video además video grabado
const DEV = DEV_PARAM !== null;
if (DEV) document.body.classList.add('dev');
if (DEV_PARAM === 'video') document.body.classList.add('devvideo');

const $ = id => document.getElementById(id);
const el = {
  video: $('video'), overlay: $('overlay'), wrap: document.querySelector('.video-wrap'),
  btnStart: $('btnStart'), btnStop: $('btnStop'), btnRecal: $('btnRecal'), btnMute: $('btnMute'), btnCsv: $('btnCsv'), btnNew: $('btnNew'), btnShare: $('btnShare'),
  fileInput: $('fileInput'), chkCnn: $('chkCnn'), chkSound: $('chkSound'), chkPoints: $('chkPoints'),
  levelText: $('levelText'), levelSub: $('levelSub'), eyesChip: $('eyesChip'),
  calibNum: $('calibNum'), calibMsg: $('calibMsg'), calibBar: $('calibBar'),
  statusText: $('statusText'), fpsChip: $('fpsChip'), fps: $('fps'), loadMsg: $('loadMsg'), toast: $('toast'),
  mPerclos: $('mPerclos'), barPerclos: $('barPerclos'), mEar: $('mEar'), mCnn: $('mCnn'), mMar: $('mMar'), mYawns: $('mYawns'),
  mPitch: $('mPitch'), mHw: $('mHw'), mYaw: $('mYaw'), mRoll: $('mRoll'),
};

// ---------------------------------------------------------------- utilidades
const fmt = (v, d = 2, sign = false) => (v === null || v === undefined || Number.isNaN(v)) ? '—'
  : ((sign && v >= 0 ? '+' : '') + v.toFixed(d)).replace('.', ',');
const median = a => { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
class MovingAvg {
  constructor(n = SMOOTH_N) { this.n = n; this.q = []; }
  push(v) { this.q.push(v); if (this.q.length > this.n) this.q.shift(); return this.q.reduce((a, b) => a + b, 0) / this.q.length; }
  clear() { this.q = []; }
}
function toast(msg, ms = 6000) {
  el.toast.textContent = msg; el.toast.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => { el.toast.hidden = true; }, ms);
}
const setState = s => { document.body.dataset.state = s; };
const setLevelAttr = l => { document.body.dataset.level = l; };

// ---------------------------------------------------------------- estado global
let landmarker = null, cnn = null, cnnOk = false;
let stream = null, isFile = false, running = false, rafId = 0, wakeLock = null;
let t0 = 0, lastTs = -1, lastFrameT = 0, fpsAvg = 0, frameCount = 0, lastP = null, cnnBusy = false;
let delegateUsed = '?', lastStatsUpdate = 0;
const stats = { detect: 0, cnn: 0, total: 0 };      // tiempos medios en ms (diagnóstico)
const ema = (avg, v) => (avg ? 0.9 * avg + 0.1 * v : v);
let base = null;                                       // calibración: {ear, mar, pitch, yaw, roll, hw}
let calib = { ear: [], mar: [], pitch: [], yaw: [], roll: [], hw: [] };
let engine = new RuleEngine(), cfg = { ...DEFAULT_CONFIG };
let sm = newSmoothers();
let log = [];
let escenario = '';                                    // prueba en curso (A-F o descanso), solo en ?dev=1
const SCEN_NAMES = { A: 'A · Normal', B: 'B · Ojos cerrados', C: 'C · Movimientos', D: 'D · Bostezos + ojos cerrados', E: 'E · Cabeceo', F: 'F · Giro de cabeza', descanso: 'Descanso' };

function newSmoothers() { return { ear: new MovingAvg(), mar: new MovingAvg(), pitch: new MovingAvg(), yaw: new MovingAvg(), roll: new MovingAvg(), hw: new MovingAvg(), cnn: new MovingAvg() }; }

// ---------------------------------------------------------------- alarma (sin archivos de audio)
let audioCtx = null, alarmTimer = null, muted = false;
function ensureAudio() {
  if (!audioCtx) { try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch { audioCtx = null; } }
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
}
function beep() {
  if (!audioCtx) return;
  const t = audioCtx.currentTime, o = audioCtx.createOscillator(), g = audioCtx.createGain();
  o.type = 'square'; o.frequency.setValueAtTime(880, t); o.frequency.setValueAtTime(660, t + 0.18);
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.35, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.4);
  o.connect(g).connect(audioCtx.destination); o.start(t); o.stop(t + 0.42);
}
function updateAlarm(level) {
  const active = level === ALERTA && el.chkSound.checked && !muted && running;
  if (active && !alarmTimer) { beep(); alarmTimer = setInterval(beep, 900); }
  if (!active && alarmTimer) { clearInterval(alarmTimer); alarmTimer = null; }
  el.btnMute.hidden = !(level === ALERTA && running && el.chkSound.checked);
  if (level !== ALERTA) muted = false;
  if (navigator.vibrate && level === ALERTA && !muted) { try { navigator.vibrate(200); } catch { /* sin vibración */ } }
}

// ---------------------------------------------------------------- carga de modelos
async function loadAssets() {
  try {
    el.loadMsg.textContent = 'Cargando clasificador de ojos…';
    if (!window.tf) throw new Error('No se pudo cargar TensorFlow.js (revise la conexión a internet).');
    await tf.ready();
    const [meta, ref, wbuf] = await Promise.all([
      fetch('models/modelo_meta.json').then(r => r.json()),
      fetch('models/referencia.json').then(r => r.json()),
      fetch('models/pesos.bin').then(r => r.arrayBuffer()),
    ]);
    cnn = buildModel(tf, wbuf, meta);
    const test = selfTest(tf, cnn, ref);
    cnnOk = test.ok;
    console.log('Autotest CNN:', test);
    if (!test.ok) toast('El modelo cargado no coincide con la referencia. Se usará solo el EAR.');

    el.loadMsg.textContent = 'Cargando detector facial…';
    const vision = await import(`${MP_BASE}/vision_bundle.mjs`);
    const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    const options = delegate => ({
      baseOptions: { modelAssetPath: 'models/face_landmarker.task', delegate },
      runningMode: 'VIDEO', numFaces: 1, outputFacialTransformationMatrixes: true,
    });
    try { landmarker = await vision.FaceLandmarker.createFromOptions(fileset, options('GPU')); delegateUsed = 'GPU'; }
    catch (e) { console.warn('GPU no disponible, se usa CPU:', e); landmarker = await vision.FaceLandmarker.createFromOptions(fileset, options('CPU')); delegateUsed = 'CPU'; }

    // calentamiento: compila los programas de la GPU ahora, para que no haya un tirón al terminar la calibración
    el.loadMsg.textContent = 'Preparando…';
    if (cnnOk) {
      const warm = document.createElement('canvas'); warm.width = 640; warm.height = 480;
      const box = [0.3, 0.3, 0.6, 0.6];
      for (let i = 0; i < 2; i++) await predictFromFrame(tf, cnn, warm, [box, box]);
    }

    el.btnStart.disabled = false; el.btnStart.textContent = '▶ Iniciar monitoreo';
    el.fileInput.disabled = false;
    el.loadMsg.textContent = cnnOk ? (DEV ? 'Modelos listos (clasificador verificado ✓)' : 'Modelos listos ✓') : 'Modelos listos (sin CNN)';
    if (!cnnOk) el.chkCnn.checked = false;
  } catch (e) {
    console.error(e);
    el.btnStart.textContent = 'No se pudo cargar';
    el.loadMsg.textContent = 'Error al cargar los modelos: ' + e.message;
    toast('No se pudieron cargar los modelos. Revise la conexión a internet y recargue la página.');
  }
}

// ---------------------------------------------------------------- inicio / fin
async function start(file = null) {
  ensureAudio();
  isFile = !!file;
  try {
    if (file) {
      el.video.srcObject = null; el.video.src = URL.createObjectURL(file); el.video.loop = false;
      el.wrap.classList.remove('mirror');
    } else {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
      el.video.srcObject = stream; el.wrap.classList.add('mirror');
    }
    await el.video.play();
  } catch (e) {
    console.error(e);
    toast(file ? 'No se pudo abrir el video.' : 'No se pudo acceder a la cámara. Permita el acceso y use HTTPS o localhost.');
    return;
  }
  await keepScreenOn(!!file);

  log = []; el.btnCsv.disabled = true; el.btnRecal.disabled = isFile;
  lastTs = -1; frameCount = 0; lastP = null; fpsAvg = 0; lastFrameT = performance.now();
  running = true; beginCalibration();
  el.fpsChip.hidden = false;
  loop();
}

function beginCalibration() {
  clearScenario();
  base = null; calib = { ear: [], mar: [], pitch: [], yaw: [], roll: [], hw: [] };
  sm = newSmoothers(); engine = new RuleEngine(cfg); muted = false;
  t0 = performance.now(); setState('calib'); setLevelAttr(NORMAL);
  el.statusText.textContent = 'Calibrando';
  el.levelText.textContent = 'CALIBRANDO'; el.levelSub.textContent = 'Mire al frente, ojos abiertos, boca cerrada';
  clearMetrics();
}

function stop(ended = false) {
  running = false; cancelAnimationFrame(rafId);
  if (el.video.cancelVideoFrameCallback && rafId) try { el.video.cancelVideoFrameCallback(rafId); } catch { /* nada */ }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  el.video.pause(); el.video.srcObject = null;
  if (isFile && el.video.src) { URL.revokeObjectURL(el.video.src); el.video.removeAttribute('src'); el.video.load(); }
  if (wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; }
  $('wakeChip').hidden = true;
  updateAlarm(NORMAL);
  const ctx = el.overlay.getContext('2d'); ctx.clearRect(0, 0, el.overlay.width, el.overlay.height);
  el.fpsChip.hidden = true; el.statusText.textContent = 'Detenido';
  el.btnCsv.disabled = log.length === 0; el.btnMute.hidden = true;
  if (log.length) {                                  // pantalla de fin de sesión, con el botón para descargar el registro
    setState('end'); setLevelAttr(NORMAL);
    el.levelText.textContent = ended ? 'FIN DEL VIDEO' : 'MONITOREO DETENIDO';
    el.levelSub.textContent = `${log.length} cuadros registrados · descargue el registro CSV`;
    window.scrollTo(0, 0);
  } else {
    setState('idle');
    toast('Se detuvo antes de terminar la calibración: no hay registro para descargar.');
  }
}

// Mantiene la pantalla encendida durante el monitoreo (Screen Wake Lock), aunque el apagado automático esté configurado
async function keepScreenOn(fromFile) {
  const chip = $('wakeChip');
  if (fromFile) { chip.hidden = true; return; }
  try {
    if (!('wakeLock' in navigator)) throw new Error('sin soporte');
    wakeLock = await navigator.wakeLock.request('screen');
    wakeLock.addEventListener('release', () => { wakeLock = null; chip.hidden = true; });
    chip.hidden = false;
  } catch (e) {
    wakeLock = null; chip.hidden = true;
    if (!keepScreenOn.avisado) {
      keepScreenOn.avisado = true;
      toast('Este navegador no puede mantener la pantalla encendida. Desactive el apagado automático del dispositivo durante el monitoreo.', 9000);
    }
  }
}

// ---------------------------------------------------------------- bucle principal
function loop() {
  if (!running) return;
  const next = () => {
    if (!running) return;
    if (el.video.requestVideoFrameCallback) rafId = el.video.requestVideoFrameCallback(onFrame);
    else rafId = requestAnimationFrame(onFrame);
  };
  const onFrame = () => { try { processFrame(); } catch (e) { console.error(e); } next(); };
  next();
}

function processFrame() {
  const v = el.video;
  if (v.readyState < 2 || !v.videoWidth) return;
  if (isFile && v.ended) { stop(true); return; }

  const nowMs = performance.now();
  const tFrame0 = nowMs;
  const dt = nowMs - lastFrameT; lastFrameT = nowMs;
  fpsAvg = fpsAvg ? 0.9 * fpsAvg + 0.1 * (1000 / Math.max(dt, 1)) : 1000 / Math.max(dt, 1);
  frameCount++;

  const tsMs = Math.max(Math.round(isFile ? v.currentTime * 1000 : nowMs), lastTs + 1); lastTs = tsMs;
  const t = isFile ? v.currentTime : (nowMs - t0) / 1000;
  const result = landmarker.detectForVideo(v, tsMs);
  stats.detect = ema(stats.detect, performance.now() - nowMs);
  const f = extractFeatures(result, v.videoWidth, v.videoHeight);
  drawOverlay(f, v.videoWidth, v.videoHeight);

  let level = NORMAL, sig = {}, info = {};
  if (base === null) {                                               // --- calibración
    const left = Math.max(0, CALIB_S - t);
    if (f) {
      calib.ear.push(f.ear); calib.mar.push(f.mar); calib.hw.push(f.hw); calib.roll.push(f.rollEye);
      if (f.pitch !== null) { calib.pitch.push(f.pitch); calib.yaw.push(f.yaw); }
    }
    el.calibNum.textContent = Math.ceil(left);
    el.calibBar.style.width = `${Math.min(100, (t / CALIB_S) * 100)}%`;
    el.calibMsg.textContent = f ? 'Mire al frente, ojos abiertos, boca cerrada' : 'No se detecta el rostro: acérquese y mire a la cámara';
    if (t >= CALIB_S && calib.ear.length >= MIN_CALIB_FRAMES) {
      base = { ear: median(calib.ear), mar: median(calib.mar), hw: median(calib.hw), roll: median(calib.roll),
               pitch: calib.pitch.length ? median(calib.pitch) : 0, yaw: calib.yaw.length ? median(calib.yaw) : 0 };
      console.log('Calibración lista:', base);
      sm = newSmoothers(); engine = new RuleEngine(cfg);
      setState('run'); el.statusText.textContent = 'Monitoreando';
    }
  } else if (!f) {                                                   // --- sin rostro
    level = engine.update(t, false); info = engine.info;
  } else {                                                           // --- decisión
    const ear = sm.ear.push(f.ear), mar = sm.mar.push(f.mar);
    const earRel = ear / base.ear, marDelta = mar - base.mar;
    const hwRel = sm.hw.push(f.hw / base.hw);
    const rollDev = sm.roll.push(angleDiff(f.rollEye, base.roll));
    const pitchDev = f.pitch !== null ? sm.pitch.push(angleDiff(f.pitch, base.pitch)) : 0;
    const yawDev = f.yaw !== null ? sm.yaw.push(angleDiff(f.yaw, base.yaw)) : 0;

    let pCnn = null;
    if (cnnOk && el.chkCnn.checked) {
      scheduleCnn(f, v);                              // asíncrono: se usa el último resultado disponible
      if (lastP !== null) pCnn = sm.cnn.push(lastP);
    }
    const { score, closed } = decideEyeClosed(earRel, pCnn, cfg);
    level = engine.update(t, true, { eyeClosed: closed, marDelta, pitchDev, hwRel, rollDev, yawDev });
    info = engine.info;
    sig = { earRel, pCnn, score, closed, marDelta, pitchDev, hwRel, rollDev, yawDev };
    log.push([t, level, 1, ear, earRel, pCnn, score, closed ? 1 : 0, marDelta, pitchDev, yawDev, rollDev, hwRel, info.perclos ?? 0, info.yawns ?? 0, escenario]);
  }
  if (base !== null) {
    if (!f) log.push([t, level, 0, '', '', '', '', '', '', '', '', '', '', info.perclos ?? 0, info.yawns ?? 0, escenario]);
    showResult(level, info, sig, f);
    updateAlarm(level);
  }
  el.fps.textContent = Math.round(fpsAvg);
  stats.total = ema(stats.total, performance.now() - tFrame0);
  if (nowMs - lastStatsUpdate > 500) { lastStatsUpdate = nowMs; showStats(); }
}

// CNN sobre los recortes de ambos ojos, sin bloquear el bucle de video
function scheduleCnn(f, v) {
  if (cnnBusy) return;
  const boxes = [eyeBox(f.rightEye, v.videoWidth, v.videoHeight), eyeBox(f.leftEye, v.videoWidth, v.videoHeight)].filter(Boolean);
  if (!boxes.length) return;
  cnnBusy = true;
  const t1 = performance.now();
  predictFromFrame(tf, cnn, v, boxes)
    .then(ps => { if (ps.length) lastP = ps.reduce((x, y) => x + y, 0) / ps.length; stats.cnn = ema(stats.cnn, performance.now() - t1); })
    .catch(e => console.error(e))
    .finally(() => { cnnBusy = false; });
}

function showStats() {
  if (!DEV) return;
  $('devStats').textContent = `Detector facial ${stats.detect.toFixed(0)} ms (${delegateUsed}) · CNN ${stats.cnn.toFixed(0)} ms · cuadro ${stats.total.toFixed(0)} ms · TF.js ${tf.getBackend()}`;
}

// ---------------------------------------------------------------- presentación
const SUB = { [NORMAL]: 'Sin señales de fatiga', [ATENCION]: 'Indicadores de cansancio', [RIESGO]: 'Varias señales de fatiga', [SIN_ROSTRO]: 'No se detecta el rostro' };
function reasonText(level, info) {
  if (level === ALERTA) {
    if (info.cabeza_caida) return 'Cabeza caída: el rostro se perdió tras inclinarse';
    if (info.microsleep) return info.fatigue_ctx ? 'Cierre ocular con fatiga acumulada' : 'Cierre ocular prolongado';
    return 'Fatiga sostenida: varios indicadores a la vez';
  }
  const r = [];
  if (info.ind_perclos) r.push('Ojos cerrados con frecuencia');
  if (info.ind_yawn) r.push('Bostezos repetidos');
  if (info.head_nod) r.push('Cabeceo');
  if (info.tilt) r.push('Cabeza inclinada');
  if (info.look_side) r.push('Mirada lateral sostenida');
  return r.length ? r.join(' · ') : SUB[level];
}
function showResult(level, info, sig, f) {
  setLevelAttr(level);
  el.levelText.textContent = level === ALERTA ? '⚠ ALERTA' : level === SIN_ROSTRO ? 'SIN ROSTRO' : level === ATENCION ? 'ATENCIÓN' : level;
  el.levelSub.textContent = reasonText(level, info);
  el.statusText.textContent = level === SIN_ROSTRO ? 'Sin rostro' : 'Monitoreando';
  const perclos = info.perclos ?? 0;
  el.mPerclos.textContent = fmt(perclos); el.barPerclos.style.width = `${Math.min(100, perclos * 100)}%`;
  el.barPerclos.style.background = perclos >= cfg.perclos_thr ? 'var(--alert)' : 'var(--ok)';
  el.mYawns.textContent = info.yawns ?? 0;
  if (!f) { el.eyesChip.textContent = 'Ojos: —'; el.eyesChip.classList.remove('closed'); return; }
  el.mEar.textContent = fmt(sig.earRel); el.mCnn.textContent = sig.pCnn === null || sig.pCnn === undefined ? '(sin CNN)' : fmt(sig.pCnn);
  el.mMar.textContent = fmt(sig.marDelta, 2, true);
  el.mPitch.textContent = fmt(sig.pitchDev, 0, true) + '°'; el.mHw.textContent = fmt(sig.hwRel);
  el.mYaw.textContent = fmt(sig.yawDev, 0, true) + '°'; el.mRoll.textContent = fmt(sig.rollDev, 0, true) + '°';
  el.mPitch.classList.toggle('hot', Math.abs(sig.pitchDev) >= cfg.pitch_dev_thr);
  el.mHw.classList.toggle('hot', sig.hwRel <= cfg.hw_nod_thr);
  el.mYaw.classList.toggle('hot', Math.abs(sig.yawDev) >= cfg.yaw_dev_thr);
  el.mRoll.classList.toggle('hot', Math.abs(sig.rollDev) >= cfg.roll_dev_thr);
  el.eyesChip.textContent = sig.closed ? 'Ojos: CERRADOS' : 'Ojos: ABIERTOS';
  el.eyesChip.classList.toggle('closed', sig.closed);
}
function clearMetrics() {
  for (const k of ['mEar', 'mCnn', 'mMar', 'mPitch', 'mHw', 'mYaw', 'mRoll']) el[k].textContent = '—';
  el.mPerclos.textContent = '0,00'; el.barPerclos.style.width = '0'; el.mYawns.textContent = '0';
  el.eyesChip.textContent = 'Ojos: —'; el.eyesChip.classList.remove('closed');
}

const COLORS = { eye: '#4ade80', eyeClosed: '#f87171', mouth: '#fbbf24', posture: '#38bdf8' };
function drawOverlay(f, w, h) {
  const c = el.overlay;
  const cw = c.clientWidth, ch = c.clientHeight, dpr = window.devicePixelRatio || 1;
  if (!cw || !ch) return;
  const pw = Math.round(cw * dpr), ph = Math.round(ch * dpr);
  if (c.width !== pw || c.height !== ph) { c.width = pw; c.height = ph; }
  const ctx = c.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, pw, ph);
  if (!f || !el.chkPoints.checked) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const { s, ox, oy } = coverTransform(w, h, cw, ch);          // el video se muestra recortado (object-fit: cover)
  const P = ([x, y]) => [x * s + ox, y * s + oy];
  const dots = (pts, color, r) => { ctx.fillStyle = color; for (const p of pts) { const [x, y] = P(p); ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); } };
  const closed = el.eyesChip.classList.contains('closed');
  dots([...f.rightEye, ...f.leftEye], closed ? COLORS.eyeClosed : COLORS.eye, 2.5);
  dots(f.mouth, COLORS.mouth, 2.5);
  const p = f.posture;
  ctx.strokeStyle = COLORS.posture; ctx.lineWidth = 1.5; ctx.setLineDash([6, 5]);
  ctx.beginPath(); ctx.moveTo(...P(p.forehead)); ctx.lineTo(...P(p.chin)); ctx.moveTo(...P(p.cheekR)); ctx.lineTo(...P(p.cheekL)); ctx.stroke(); ctx.setLineDash([]);
  dots([p.forehead, p.chin, p.cheekR, p.cheekL, p.nose], COLORS.posture, 4);
}

// ---------------------------------------------------------------- CSV
function buildCsv() {
  const head = ['t_s', 'nivel', 'rostro', 'ear', 'ear_rel', 'p_cnn', 'puntaje_cierre', 'ojo_cerrado', 'mar_delta', 'pitch_dev', 'yaw_dev', 'roll_dev', 'hw_rel', 'perclos', 'bostezos', 'escenario'];
  const cell = v => (typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : v ?? '');
  const csv = '\ufeff' + [head, ...log].map(r => r.map(cell).join(',')).join('\r\n');
  const name = `vigileye_sesion_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.csv`;
  return { csv, name };
}
function downloadCsv() {
  const { csv, name } = buildCsv();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60000);
  toast('Registro descargado. Búsquelo en la carpeta Descargas del dispositivo.', 8000);
}
async function shareCsv() {                       // en el celular permite enviarlo directo (WhatsApp, correo, Drive…)
  const { csv, name } = buildCsv();
  const file = new File([csv], name, { type: 'text/csv' });
  try { await navigator.share({ files: [file], title: 'Registro VigilEye' }); }
  catch (e) { if (e.name !== 'AbortError') toast('No se pudo compartir. Use "Descargar registro (CSV)".'); }
}
function setScenario(s) {
  if (base === null) { toast('Espere a que termine la calibración (10 s) para marcar una prueba.'); return; }
  escenario = s;
  engine = new RuleEngine(cfg); sm = newSmoothers();     // reinicia PERCLOS y bostezos: cada prueba parte limpia
  $('scenNow').textContent = SCEN_NAMES[s] || s;
  document.querySelectorAll('#scenBar button').forEach(b => b.classList.toggle('on', b.dataset.s === s));
}
function clearScenario() {
  escenario = ''; $('scenNow').textContent = 'ninguna';
  document.querySelectorAll('#scenBar button').forEach(b => b.classList.remove('on'));
}
function newSession() { clearScenario(); log = []; el.btnCsv.disabled = true; setState('idle'); setLevelAttr(NORMAL); window.scrollTo(0, 0); }
if (DEV) window.vigil = { get log() { return log; }, buildCsv, downloadCsv, newSession, stop, setScenario, testSetBase() { base = { ear: 0.3, mar: 0.3, hw: 1, roll: 0, pitch: 0, yaw: 0 }; } };

// ---------------------------------------------------------------- eventos
el.btnStart.addEventListener('click', () => start());
el.fileInput.addEventListener('change', e => { const f = e.target.files[0]; if (f) start(f); e.target.value = ''; });
el.btnStop.addEventListener('click', () => stop(false));
el.btnRecal.addEventListener('click', () => { if (running && !isFile) beginCalibration(); });
el.btnMute.addEventListener('click', () => { muted = true; updateAlarm(ALERTA); el.btnMute.hidden = true; });
el.btnCsv.addEventListener('click', downloadCsv);
document.querySelectorAll('#scenBar button').forEach(b => b.addEventListener('click', () => setScenario(b.dataset.s)));
el.btnShare.addEventListener('click', shareCsv);
el.btnNew.addEventListener('click', newSession);
try {                                             // mostrar "Compartir" solo si el navegador puede compartir archivos
  if (navigator.canShare && navigator.canShare({ files: [new File(['a'], 'a.csv', { type: 'text/csv' })] })) el.btnShare.hidden = false;
} catch { /* sin compartir */ }
if (/FBAN|FBAV|Instagram|WhatsApp|Line\/|MicroMessenger|Messenger|TikTok/i.test(navigator.userAgent)) {
  toast('Está en un navegador interno de una aplicación: abra esta página en Chrome o Safari para poder descargar el registro.', 12000);
}
el.chkSound.addEventListener('change', () => { if (el.chkSound.checked) ensureAudio(); updateAlarm(engine.level); });
el.video.addEventListener('ended', () => { if (running && isFile) stop(true); });
document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState === 'visible' && running && !isFile && !wakeLock) await keepScreenOn(false);
});

loadAssets();
