/**
 * rules.js - Motor de reglas de VigilEye (puerto a JavaScript de rules.py, con postura de la cabeza ampliada).
 *
 * Recibe, cuadro a cuadro, señales ya procesadas (ojo cerrado sí/no, delta de MAR, postura de la cabeza)
 * y devuelve el nivel: NORMAL, ATENCION, RIESGO, ALERTA (o SIN_ROSTRO).
 * No depende del DOM ni de MediaPipe, por lo que se puede probar con datos simulados.
 *
 * Los valores de DEFAULT_CONFIG son valores iniciales: se ajustan con las pruebas en uso real.
 */
export const NORMAL = 'NORMAL', ATENCION = 'ATENCION', RIESGO = 'RIESGO', ALERTA = 'ALERTA', SIN_ROSTRO = 'SIN_ROSTRO';

export const DEFAULT_CONFIG = {
  // Estado del ojo (fusión EAR + CNN)
  w_ear: 0.6,                // peso del puntaje basado en EAR
  w_cnn: 0.4,                // peso de la probabilidad de la CNN
  closed_threshold: 0.5,     // puntaje fusionado >= esto => ojo cerrado
  cnn_veto_thr: 0.3,         // si la CNN está disponible, debe dar al menos esto para confirmar un cierre
  // PERCLOS
  perclos_window_s: 60.0,
  perclos_thr: 0.15,         // fracción de tiempo con ojos cerrados
  perclos_min_s: 10.0,       // no se evalúa hasta tener esta cobertura de datos
  // Microsueño (cierre continuo de los ojos): el tiempo exigido depende del contexto de fatiga
  microsleep_s: 4.0,         // sin otras señales de fatiga: evita falsas alarmas por movimientos
  microsleep_ctx_s: 2.0,     // con fatiga acumulada: ALERTA inmediata
  fatigue_memory_s: 60.0,    // cuánto tiempo se recuerda un cabeceo para el contexto de fatiga
  // Bostezo (MAR relativo a la calibración)
  mar_delta_thr: 0.40,
  yawn_min_s: 1.5,
  yawn_window_s: 60.0,
  yawn_count_thr: 2,         // bostezos en la ventana para que cuenten como señal (uno solo no es señal)
  // Cabeceo (cabeza hacia adelante/abajo): ángulo de pitch O proporción alto/ancho del rostro reducida
  pitch_dev_thr: 15.0,       // grados respecto a la postura de calibración
  hw_nod_thr: 0.80,          // (alto/ancho del rostro) relativo a la calibración; baja al cabecear
  head_hold_s: 1.5,
  // Inclinación lateral (oreja hacia el hombro)
  roll_dev_thr: 25.0,        // grados respecto a la calibración
  roll_hold_s: 2.0,
  // Giro lateral sostenido (mirar de lado): solo nivel de ATENCIÓN; mirar el retrovisor es normal
  yaw_dev_thr: 40.0,
  yaw_hold_s: 3.0,
  // Escalado de niveles
  risk_to_alert_s: 5.0,      // RIESGO sostenido => ALERTA
  no_face_s: 3.0,            // tiempo sin rostro para mostrar SIN_ROSTRO
  // Cabeza caída: rostro perdido cuando la cabeza ya venía inclinada hacia adelante
  head_down_pitch: 12.0,     // grados de desviación antes de perder el rostro
  head_down_hw: 0.88,        // o proporción alto/ancho por debajo de este valor
  head_down_lost_s: 2.0,     // segundos sin rostro => ALERTA
};

/** Puntaje de cierre en [0, 1]. earRel = EAR / EAR_reposo. Si pCnn es null se usa solo el EAR. */
export function eyeClosedScore(earRel, pCnn, cfg = DEFAULT_CONFIG) {
  const sEar = Math.min(1.0, Math.max(0.0, (1.0 - earRel) / 0.5));
  if (pCnn === null || pCnn === undefined) return sEar;
  return cfg.w_ear * sEar + cfg.w_cnn * Number(pCnn);
}

/**
 * Decide si el ojo está cerrado. Con CNN, el EAR y la red deben coincidir: el puntaje fusionado debe superar el umbral
 * Y la CNN debe dar al menos cnn_veto_thr (evita falsos cierres por movimientos de cabeza o mirada hacia abajo).
 * Sin CNN se usa solo el EAR.
 */
export function decideEyeClosed(earRel, pCnn, cfg = DEFAULT_CONFIG) {
  const score = eyeClosedScore(earRel, pCnn, cfg);
  let closed = score >= cfg.closed_threshold;
  if (closed && pCnn !== null && pCnn !== undefined && pCnn < cfg.cnn_veto_thr) closed = false;
  return { score, closed };
}

export class RuleEngine {
  constructor(cfg = DEFAULT_CONFIG) {
    this.cfg = { ...DEFAULT_CONFIG, ...cfg };
    this.eyeSamples = [];      // [t, cerrado 0/1] para PERCLOS
    this.yawnTimes = [];       // instantes en que se contó un bostezo
    this.closedSince = null;
    this.mouthOpenSince = null;
    this.yawnCounted = false;
    this.headOffSince = null;
    this.tiltSince = null;
    this.sideSince = null;
    this.riskSince = null;
    this.lastFaceT = null;
    this.lastWasNod = false;
    this.lastNodT = null;      // último instante con cabeceo sostenido
    this.level = NORMAL;
    this.info = {};
  }

  /**
   * @param {number} t tiempo en segundos
   * @param {boolean} faceFound
   * @param {{eyeClosed?:boolean, marDelta?:number, pitchDev?:number, hwRel?:number, rollDev?:number, yawDev?:number}} s
   */
  update(t, faceFound, { eyeClosed = false, marDelta = 0.0, pitchDev = 0.0, hwRel = 1.0, rollDev = 0.0, yawDev = 0.0 } = {}) {
    const c = this.cfg;
    if (faceFound) {
      this.lastFaceT = t;
      this.lastWasNod = Math.abs(pitchDev) >= c.head_down_pitch || hwRel <= c.head_down_hw;
    } else {
      const lost = this.lastFaceT === null ? Infinity : t - this.lastFaceT;
      // Cabeza caída: la cabeza ya iba hacia adelante y el rostro dejó de verse
      if (this.lastFaceT !== null && this.lastWasNod && lost >= c.head_down_lost_s) {
        this.level = ALERTA;
        this.info = { perclos: this._perclos(), microsleep: false, cabeza_caida: true };
        return this.level;
      }
      if (this.lastFaceT === null || lost >= c.no_face_s) {
        this._resetTimers();
        this.level = SIN_ROSTRO;
        this.info = { perclos: this._perclos(), microsleep: false };
        return this.level;
      }
      return this.level;      // rostro perdido por poco tiempo: se mantiene el nivel anterior
    }

    // --- ojos: cierre continuo y PERCLOS
    if (eyeClosed) {
      if (this.closedSince === null) this.closedSince = t;
    } else {
      this.closedSince = null;
    }
    const closedDur = this.closedSince === null ? 0.0 : t - this.closedSince;
    this.eyeSamples.push([t, eyeClosed ? 1 : 0]);
    while (this.eyeSamples.length && t - this.eyeSamples[0][0] > c.perclos_window_s) this.eyeSamples.shift();
    const perclos = this._perclos();

    // --- bostezo: boca muy abierta de forma sostenida
    if (marDelta >= c.mar_delta_thr) {
      if (this.mouthOpenSince === null) this.mouthOpenSince = t;
      if (!this.yawnCounted && t - this.mouthOpenSince >= c.yawn_min_s) {
        this.yawnTimes.push(t);
        this.yawnCounted = true;
      }
    } else {
      this.mouthOpenSince = null;
      this.yawnCounted = false;
    }
    while (this.yawnTimes.length && t - this.yawnTimes[0] > c.yawn_window_s) this.yawnTimes.shift();

    // --- postura de la cabeza
    const nodNow = Math.abs(pitchDev) >= c.pitch_dev_thr || hwRel <= c.hw_nod_thr;
    this.headOffSince = this._hold(this.headOffSince, nodNow, t);
    const headNod = this.headOffSince !== null && t - this.headOffSince >= c.head_hold_s;
    if (headNod) this.lastNodT = t;

    this.tiltSince = this._hold(this.tiltSince, Math.abs(rollDev) >= c.roll_dev_thr, t);
    const tilt = this.tiltSince !== null && t - this.tiltSince >= c.roll_hold_s;

    this.sideSince = this._hold(this.sideSince, Math.abs(yawDev) >= c.yaw_dev_thr, t);
    const lookSide = this.sideSince !== null && t - this.sideSince >= c.yaw_hold_s;

    // --- indicadores y nivel
    const indPerclos = perclos >= c.perclos_thr;
    const indYawn = this.yawnTimes.length >= c.yawn_count_thr;
    const nActive = Number(indPerclos) + Number(indYawn) + Number(headNod) + Number(tilt);
    // contexto de fatiga: bostezos repetidos, PERCLOS alto o cabeceo reciente => el microsueño se confirma antes
    // el PERCLOS del contexto excluye el cierre que está en curso (si no, un cierre aislado se autoconfirmaría)
    const perclosPrev = this.closedSince === null ? perclos : this._perclos(this.closedSince);
    const fatigueCtx = indYawn || perclosPrev >= c.perclos_thr || headNod || (this.lastNodT !== null && t - this.lastNodT <= c.fatigue_memory_s);
    const microsleepThr = fatigueCtx ? c.microsleep_ctx_s : c.microsleep_s;
    const microsleep = closedDur >= microsleepThr;

    let level;
    if (microsleep) {
      level = ALERTA;
    } else if (nActive >= 2) {
      if (this.riskSince === null) this.riskSince = t;
      level = (t - this.riskSince >= c.risk_to_alert_s) ? ALERTA : RIESGO;
    } else if (nActive === 1) {
      level = ATENCION;
    } else {
      level = NORMAL;
    }
    if (nActive < 2) this.riskSince = null;
    if (level === NORMAL && lookSide) level = ATENCION;     // mirada lateral sostenida: solo atención

    this.level = level;
    this.info = { perclos, closed_dur: closedDur, yawns: this.yawnTimes.length, head_nod: headNod, tilt,
                  look_side: lookSide, microsleep, n_active: nActive,
                  ind_perclos: indPerclos, ind_yawn: indYawn, fatigue_ctx: fatigueCtx, microsleep_thr: microsleepThr };
    return level;
  }

  _hold(since, cond, t) { return cond ? (since === null ? t : since) : null; }

  /** PERCLOS: fracción de tiempo con ojos cerrados. Con `cut` solo se usan las muestras anteriores a ese instante. */
  _perclos(cut = Infinity) {
    const samples = cut === Infinity ? this.eyeSamples : this.eyeSamples.filter(([t]) => t < cut);
    if (!samples.length) return 0.0;
    const first = samples[0][0], last = samples[samples.length - 1][0];
    if (last - first < this.cfg.perclos_min_s) return 0.0;   // pocas muestras: aún no es confiable
    let s = 0;
    for (const [, v] of samples) s += v;
    return s / samples.length;
  }

  _resetTimers() {
    this.closedSince = null;
    this.mouthOpenSince = null;
    this.yawnCounted = false;
    this.headOffSince = null;
    this.tiltSince = null;
    this.sideSince = null;
    this.riskSince = null;
  }
}
