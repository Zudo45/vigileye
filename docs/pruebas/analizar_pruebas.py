"""
Análisis de los registros CSV de las pruebas de VigilEye (sección 6.2 del informe).

Uso (desde esta carpeta, con Python 3.9 o superior, sin bibliotecas adicionales):
    python analizar_pruebas.py

Lee participante1_laptop.csv, participante2_laptop.csv y participante3_celular.csv y calcula:
  - tasa de procesamiento (FPS) a partir de los tiempos de cada cuadro,
  - proporción de cuadros por nivel y episodios de ALERTA en cada prueba (A a F),
  - tiempo de respuesta ante un cierre ocular continuo,
  - episodios de ALERTA en las pruebas donde no se esperaba alarma (A, C y F),
  - probabilidad media de la CNN con ojo abierto y con cierre confirmado, y cuadros descartados por el filtro de la CNN.
"""
import csv
import collections
import os
import statistics as st

AQUI = os.path.dirname(os.path.abspath(__file__))
ARCHIVOS = {
    'Participante 1 (computadora portátil)': 'participante1_laptop.csv',
    'Participante 2 (computadora portátil)': 'participante2_laptop.csv',
    'Participante 3 (teléfono celular)': 'participante3_celular.csv',
}
HUECO_CIERRE = 0.35        # s: interrupciones menores se consideran el mismo cierre
CIERRE_MIN = 1.0           # s: duración mínima de un cierre para analizarlo
CNN_FILTRO = 0.3           # umbral del filtro de la CNN usado por la aplicación


def num(x):
    return float(x) if x not in ('', None) else None


def cargar(ruta):
    filas = list(csv.DictReader(open(ruta, encoding='utf-8-sig')))
    for r in filas:
        r['t'] = float(r['t_s'])
        for k in ('ear_rel', 'p_cnn', 'puntaje_cierre', 'mar_delta', 'pitch_dev', 'yaw_dev', 'roll_dev', 'hw_rel', 'perclos'):
            r[k] = num(r[k])
        r['cerrado'] = r['ojo_cerrado'] == '1'
        r['rostro_i'] = int(r['rostro'])
    return filas


def tramos(filas):
    """Tramos consecutivos con la misma prueba marcada."""
    out, cur = [], None
    for r in filas:
        if cur is None or cur['e'] != r['escenario']:
            cur = {'e': r['escenario'], 'filas': []}
            out.append(cur)
        cur['filas'].append(r)
    return out


def episodios_alerta(filas):
    eps, ini = [], None
    for r in filas:
        if r['nivel'] == 'ALERTA':
            if ini is None:
                ini = r['t']
        elif ini is not None:
            eps.append((ini, r['t']))
            ini = None
    if ini is not None:
        eps.append((ini, filas[-1]['t']))
    return eps


def cierres_continuos(filas):
    runs, cur = [], None
    for r in filas:
        if r['rostro_i'] and r['cerrado']:
            if cur is None:
                cur = [r['t'], r['t']]
            elif r['t'] - cur[1] <= HUECO_CIERRE:
                cur[1] = r['t']
            else:
                runs.append(cur)
                cur = [r['t'], r['t']]
        elif cur is not None and r['t'] - cur[1] > HUECO_CIERRE:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    return [c for c in runs if c[1] - c[0] >= CIERRE_MIN]


def main():
    total_min_sin_alarma, total_ep_sin_alarma = 0.0, 0
    for nombre, archivo in ARCHIVOS.items():
        filas = cargar(os.path.join(AQUI, archivo))
        dts = [b['t'] - a['t'] for a, b in zip(filas, filas[1:]) if 0 < b['t'] - a['t'] < 2]
        q = st.quantiles(dts, n=10)
        print('\n' + '=' * 78 + '\n' + nombre)
        print(f"  cuadros: {len(filas)} | registro: {filas[-1]['t'] - filas[0]['t']:.0f} s | "
              f"FPS mediana {1 / st.median(dts):.1f} (p10 {1 / q[8]:.1f} a p90 {1 / q[0]:.1f}) | "
              f"sin rostro {100 * sum(1 for r in filas if not r['rostro_i']) / len(filas):.1f} %")

        # CNN
        ab = [r['p_cnn'] for r in filas if r['rostro_i'] and not r['cerrado'] and r['p_cnn'] is not None]
        ce = [r['p_cnn'] for r in filas if r['rostro_i'] and r['cerrado'] and r['p_cnn'] is not None]
        vet = [r for r in filas if r['rostro_i'] and not r['cerrado'] and r['puntaje_cierre'] is not None
               and r['puntaje_cierre'] >= 0.5 and r['p_cnn'] is not None and r['p_cnn'] < CNN_FILTRO]
        cand = len(vet) + sum(1 for r in filas if r['rostro_i'] and r['cerrado'])
        print(f"  CNN: prob. media ojo abierto {st.mean(ab):.2f} | con cierre confirmado {st.mean(ce):.2f} | "
              f"cuadros descartados por el filtro: {len(vet)} ({100 * len(vet) / cand:.1f} %)")

        # por prueba
        print('  Prueba | cuadros | duración | proporción por nivel | episodios de ALERTA')
        sin_alarma_s, sin_alarma_ep = 0.0, 0
        for e in sorted({r['escenario'] for r in filas}):
            seg = [s['filas'] for s in tramos(filas) if s['e'] == e]
            dur = sum(f[-1]['t'] - f[0]['t'] for f in seg if len(f) > 1)
            fil = [r for f in seg for r in f]
            niveles = collections.Counter(r['nivel'] for r in fil)
            ep = sum(len(episodios_alerta(f)) for f in seg)
            pct = ', '.join(f"{k} {100 * v / len(fil):.0f} %" for k, v in niveles.most_common())
            print(f"   {e or '(sin marcar)':>12} | {len(fil):5d} | {dur:6.1f} s | {pct} | {ep}")
            if e in ('A', 'C', 'F'):
                sin_alarma_s += dur
                sin_alarma_ep += ep
        total_min_sin_alarma += sin_alarma_s / 60
        total_ep_sin_alarma += sin_alarma_ep
        print(f"  Pruebas A, C y F: {sin_alarma_s:.1f} s, {sin_alarma_ep} episodios de ALERTA")

        # tiempo de respuesta
        print('  Cierres continuos en B y D (duración del cierre -> tiempo hasta la ALERTA):')
        for e in ('B', 'D'):
            for s in tramos(filas):
                if s['e'] != e:
                    continue
                for a, b in cierres_continuos(s['filas']):
                    al = [r['t'] for r in s['filas'] if r['nivel'] == 'ALERTA' and a - 0.05 <= r['t'] <= b + 1.0]
                    resp = f"{min(al) - a:.2f} s" if al else 'sin alerta'
                    print(f"     {e}: t={a:6.1f} s | cierre de {b - a:4.1f} s -> {resp}")

    print('\n' + '=' * 78)
    print(f"Pruebas sin alarma esperada (A, C y F), conjunto: {total_min_sin_alarma:.2f} min, "
          f"{total_ep_sin_alarma} episodios de ALERTA -> {total_ep_sin_alarma / total_min_sin_alarma:.2f} por minuto")


if __name__ == '__main__':
    main()
