# VigilEye · Detector de somnolencia y fatiga al volante

Aplicación web que detecta señales visuales de somnolencia en el conductor usando la cámara del dispositivo, y avisa con
sonido y pantalla roja. **Corre por completo en el navegador**: el video no se envía ni se guarda, y cada persona que abre
la página usa su propia cámara y su propia calibración.

> **Demostración:** `https://zudo45.github.io/vigileye/` &nbsp;·&nbsp; abrir desde el celular (requiere HTTPS, ya incluido en GitHub Pages)

Proyecto del curso **Inteligencia Artificial: Principios y Técnicas** · Ingeniería de Sistemas e Inteligencia Artificial ·
Universidad Privada Antenor Orrego (UPAO), Trujillo, 2026. Docente: Ing. Teobaldo Hernán Sagástegui Chigne.

**Integrantes:** Diaz Polo Leonardo · Flores Rodriguez Diego Alexander · Lezama Pasco Diegoandree Ronald · Ponce Díaz Luis Francesco · Rios Rios Bryam

---

## Cómo funciona

```
Cámara → MediaPipe Face Landmarker → EAR · MAR · postura de la cabeza · recorte de ojo
       → calibración de 10 s → CNN (ojo abierto/cerrado) + EAR → motor de reglas
       → NORMAL · ATENCIÓN · RIESGO · ALERTA → sonido, pantalla roja y registro CSV
```

| Señal | Cómo se mide |
|---|---|
| Ojos cerrados | EAR (relación de aspecto del ojo) fusionado con una **CNN propia** que clasifica el recorte del ojo; PERCLOS en 60 s y cierre continuo. Un cierre solo cuenta si el EAR y la CNN coinciden |
| Microsueño | Ojos cerrados seguidos: **4 s** sin otras señales de fatiga, **2 s** si ya hay fatiga acumulada |
| Bostezo | MAR sostenido respecto de la calibración. Un solo bostezo no cuenta; **2 en 60 s** activan la señal |
| Cabeceo | Ángulo de cabeza y **proporción alto/ancho del rostro** (baja al agachar la cabeza, no cambia al girar ni al alejarse) |
| Inclinación lateral | Ángulo de la línea entre las esquinas externas de los ojos |
| Giro lateral | Sostenido por más de 3 s solo da ATENCIÓN (mirar el retrovisor es normal) |
| Cabeza caída | El rostro desaparece cuando la cabeza ya venía inclinada hacia adelante: ALERTA |

**Niveles:** una señal activa = ATENCIÓN · dos o más = RIESGO · dos o más sostenidas 5 s, microsueño o cabeza caída = ALERTA.
Todos los umbrales están en [`js/rules.js`](js/rules.js) y son valores iniciales para ajustar con pruebas reales.

## Resultados del clasificador de ojos

CNN ligera (285 569 parámetros) entrenada con el [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) (84 898 imágenes de 37 personas),
con partición **por persona** 70/15/15, ponderación de clases y aumento de datos. Evaluación única con 12 927 imágenes de
11 personas no vistas (clase positiva: ojo cerrado, umbral 0,5):

| Exactitud | Sensibilidad | Especificidad | Precisión | F1 | AUC |
|---|---|---|---|---|---|
| 94,0 % | 98,5 % | 87,1 % | 92,0 % | 95,1 % | 0,987 |

Detalle, curvas y matriz de confusión en [`docs/resultados/`](docs/resultados/).

## Ejecutar en tu computadora

1. Doble clic en **`INICIAR.bat`** (requiere Python; abre un servidor local y el navegador en `http://localhost:8000`).
   Alternativa: `python -m http.server 8000` dentro de esta carpeta.
2. Pulsa **Iniciar monitoreo**, acepta el permiso de cámara y mira al frente durante la calibración (10 s).

La cámara solo funciona en `localhost` o en una página con **HTTPS**. Se necesita conexión a internet al abrir la página
(MediaPipe y TensorFlow.js se cargan desde jsDelivr).

### Opciones de desarrollo (para pruebas)
Agregando `?dev=1` a la dirección aparecen el panel de **Diagnóstico** (tiempos y FPS) y el interruptor de la CNN.
La tarjeta **Descargar registro (CSV)** guarda una fila por cuadro (nivel, EAR, probabilidad de la CNN, postura, PERCLOS…) y sirve
como evidencia de las pruebas.

## Estructura del repositorio

```
index.html · css/ · js/       aplicación web (HTML, CSS y JavaScript)
models/                       detector facial de MediaPipe y pesos de la CNN (pesos.bin)
notebooks/                    Colab: 01 preparación de datos · 02 entrenamiento, evaluación y exportación
tests/                        pruebas con Node (reglas, medidas geométricas y coincidencia de la CNN con Python)
docs/                         informe, guía de pruebas, resultados del entrenamiento y maquetas
```

| Archivo | Función |
|---|---|
| `js/app.js` | Bucle principal, calibración, alarma, registro CSV |
| `js/features.js` | EAR, MAR, ángulos de la cabeza, proporción del rostro, recorte del ojo |
| `js/rules.js` | Motor de reglas y niveles de riesgo |
| `js/cnn.js` | Clasificador de ojo abierto/cerrado en TensorFlow.js (se reconstruye desde `pesos.bin`) |

## Publicar cambios

GitHub Pages hace que los navegadores guarden los archivos 10 minutos. Para que un cambio se vea de inmediato, **subir el número de versión**
(`?v=4` → `?v=5`) en `index.html` (CSS y `app.js`) y en los `import` de `js/app.js`; después `git add .`, `git commit` y `git push`.

## Pruebas automáticas

```
npm install      # solo para la prueba de la CNN (instala @tensorflow/tfjs)
npm test         # 18 escenarios de reglas, medidas geométricas y coincidencia de la CNN con Python
```

La prueba de la CNN comprueba que el modelo en JavaScript reproduce las salidas calculadas en Python (diferencia ≈ 6·10⁻⁸).
Al abrir la página se ejecuta el mismo autotest dentro del navegador.

## Reentrenar el modelo

Los cuadernos de `notebooks/` se ejecutan en Google Colab con GPU. `01_Preparacion_Datos_MRL.ipynb` descarga y prepara los datos
(se guardan en Google Drive) y `Notebook_02_Entrenamiento_CNN_v2.ipynb` entrena, evalúa y exporta los pesos
(`pesos.bin`, `modelo_meta.json`) que usa la web.

## Limitaciones

- La CNN se entrenó con recortes de ojo infrarrojos; con cámaras a color puede variar, por eso el EAR siempre participa en la decisión.
- Los umbrales de postura, bostezo y tiempos de alerta dependen de la posición de la cámara y se ajustan con pruebas reales.
- El rendimiento (FPS) depende del dispositivo; el objetivo es 15 FPS o más.
- No es un diagnóstico médico: detecta patrones visuales asociados con una posible pérdida de alerta.

## Referencias principales

- Soukupová, T., & Čech, J. (2016). Real-time eye blink detection using facial landmarks. *21st Computer Vision Winter Workshop*.
- Fusek, R. (2018). Pupil localization using geodesic distance. *ISVC 2018* (MRL Eye Dataset).
- Lugaresi, C. et al. (2019). MediaPipe: A framework for building perception pipelines. arXiv:1906.08172.
- Dinges, D. F., & Grace, R. (1998). *PERCLOS: A valid psychophysiological measure of alertness* (FHWA-MCRT-98-006).
