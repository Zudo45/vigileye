/**
 * cnn.js - Clasificador de estado ocular (ojo abierto / cerrado) en TensorFlow.js.
 *
 * Reconstruye la CNN entrenada en el Notebook 02 v2 (64x64x1 -> 3 bloques conv+pool -> densa 64 -> sigmoide)
 * y carga sus pesos desde models/pesos.bin (float32, orden de model.get_weights()).
 * La salida es p_cerrado: probabilidad de que el ojo esté cerrado. La entrada son recortes con valores 0-255;
 * la división entre 255 se hace aquí, igual que la capa de reescalado del modelo original.
 *
 * `tf` se recibe como parámetro para poder usar el mismo código en el navegador y en las pruebas con Node.
 */
export function buildModel(tf, weightsBuffer, meta) {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({ inputShape: [64, 64, 1], filters: 16, kernelSize: 3, padding: 'same', activation: 'relu' }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu' }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: 0.3 }));
  m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  const f32 = new Float32Array(weightsBuffer);
  let off = 0;
  const tensors = meta.tensores.map(({ forma }) => {
    const n = forma.reduce((a, b) => a * b, 1);
    const t = tf.tensor(f32.slice(off, off + n), forma);
    off += n;
    return t;
  });
  if (off !== f32.length) throw new Error(`pesos.bin no coincide con el modelo (${off} de ${f32.length} valores)`);
  m.setWeights(tensors);
  tensors.forEach(t => t.dispose());
  return m;
}

/** Probabilidad de ojo cerrado para cada recorte (Float32Array de 4096 con valores 0-255). */
export function predictClosed(tf, model, crops) {
  if (!crops.length) return [];
  return tf.tidy(() => {
    const flat = new Float32Array(crops.length * 4096);
    crops.forEach((c, i) => flat.set(c, i * 4096));
    const x = tf.tensor4d(flat, [crops.length, 64, 64, 1]).div(255);
    return Array.from(model.predict(x).dataSync());
  });
}

/**
 * Probabilidad de ojo cerrado para cada caja, tomando los recortes directamente del cuadro de video.
 * Todo el procesamiento (gris, recorte, redimensionado, red) ocurre en la GPU y el resultado se lee de forma asíncrona,
 * para no bloquear el bucle de video.  boxes: [[y1, x1, y2, x2], ...] normalizadas.
 */
export async function predictFromFrame(tf, model, video, boxes) {   // `video`: <video> o <canvas>
  const out = tf.tidy(() => {
    // se recorta primero (solo 2 recortes de 64x64) y después se pasa a gris: es mucho menos trabajo que convertir todo el cuadro
    const px = tf.cast(tf.browser.fromPixels(video), 'float32').expandDims(0);                    // [1, h, w, 3]
    const crops = tf.image.cropAndResize(px, boxes, boxes.map(() => 0), [64, 64], 'bilinear');    // [n, 64, 64, 3]
    const gray = crops.mul(tf.tensor1d([0.299, 0.587, 0.114])).sum(3).expandDims(-1);             // [n, 64, 64, 1] 0-255
    return model.predict(gray.div(255));
  });
  const data = await out.data();
  out.dispose();
  return Array.from(data);
}

/** Compara la salida del modelo con los valores de referencia calculados en Python. */
export function selfTest(tf, model, ref, tol = 1e-3) {
  const crops = ref.imagenes.map(a => Float32Array.from(a));
  const out = predictClosed(tf, model, crops);
  const maxDiff = Math.max(...out.map((p, i) => Math.abs(p - ref.p_cerrado[i])));
  return { ok: maxDiff < tol, maxDiff, salida: out, esperado: ref.p_cerrado };
}
