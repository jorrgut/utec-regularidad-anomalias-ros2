#!/usr/bin/env python3
"""
Proyecto integrador: genera los registros de una auditoría.

Dos modos:
  --practica   el caso resuelto del libro: eventos FIJOS y publicados (eventos_practica.txt)
  --evaluacion un caso nuevo para evaluar: eventos AL AZAR según la semilla. Los registros
               van a <salida>/estudiantes/ y la solución a <salida>/solucion_NO_DISTRIBUIR/.

Registros que se generan (formato del logger de la Unidad 1: t_sec, v, w):
  nominal_entrenamiento.csv  10 min sin anomalías (entrenar modelos, Unidad 6)
  nominal_calibracion.csv     5 min sin anomalías, otra corrida (calibrar el detector)
  registro_auditoria.csv     15 min a auditar
  ficha_robot.txt            lo que se sabe del sistema (lo mismo que sabría un auditor)

Uso:
  python3 generar_caso.py --practica --salida caso_practica
  python3 generar_caso.py --evaluacion --semilla 20271 --salida caso_2027   # solo la cátedra
"""
import argparse
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, '..', 'u6', 'simulate_cmdvel_csv.py')
DUR = 900.0

FICHA = """FICHA DEL ROBOT
===============
Robot: TurtleBot3 burger (simulado en Gazebo), ROS 2.
Tópico registrado: /cmd_vel, con tiempo simulado (use_sim_time).
Publicador nominal: un generador a 10 Hz que repite el ciclo
  avanzar 6 s (v = 0.20 m/s) -> detener 2 s -> girar 3 s (w = 0.80 rad/s) -> detener 2 s.
Límites físicos: |v| <= 0.22 m/s, |w| <= 2.84 rad/s.
Los dos registros "nominal_*" se tomaron en operación normal, en otros días.
El registro "registro_auditoria.csv" dura 15 minutos. El operador informa que "en algún
momento el robot se comportó raro". No hay más información.
"""

# Catálogo de anomalías: (nombre, argumentos del simulador para el intervalo [a, a + d])
CATALOGO = {
    'flood':      lambda a, d: ['--flood-at', a, '--flood-len', d, '--flood-rate', 40],
    'pausa':      lambda a, d: ['--pause-at', a, '--pause-len', d],
    'replay':     lambda a, d: ['--replay-at', a, '--replay-len', d, '--replay-shift', 6.5,
                                '--replay-mode', 'replace'],
    'estructura': lambda a, d: ['--segment-mode', 'shuffle_blocks', '--segment-at', a,
                                '--segment-len', d],
    'inyeccion':  lambda a, d: ['--segment-mode', 'iid', '--segment-at', a, '--segment-len', d],
}
DURACIONES = {'flood': (8, 15), 'pausa': (1.5, 3), 'replay': (8, 15),
              'estructura': (20, 40), 'inyeccion': (10, 20)}

PRACTICA = [('estructura', 120, 30), ('flood', 300, 10), ('pausa', 450, 2),
            ('replay', 600, 10), ('inyeccion', 750, 15)]


def sim(out, *args):
    r = subprocess.run([sys.executable, SIM, '--out', out, *map(str, args)],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)


def sortear_eventos(rng):
    # "estructura" e "inyeccion" usan el mismo segmento del simulador: va uno de los dos
    tipos = ['flood', 'pausa', 'replay', rng.choice(['estructura', 'inyeccion'])]
    rng.shuffle(tipos)
    k = rng.randint(3, 4)
    tipos = tipos[:k]
    eventos, t = [], 60.0
    hueco = (DUR - 120) / k
    for tipo in tipos:
        d = round(rng.uniform(*DURACIONES[tipo]), 1)
        a = round(t + rng.uniform(0, hueco - d - 20), 1)
        eventos.append((tipo, a, d))
        t += hueco
    return eventos


def main():
    ap = argparse.ArgumentParser(description='Registros del proyecto integrador')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--practica', action='store_true')
    g.add_argument('--evaluacion', action='store_true')
    ap.add_argument('--semilla', type=int, default=None, help='obligatoria con --evaluacion')
    ap.add_argument('--salida', default='caso_practica')
    a = ap.parse_args()

    if a.practica:
        eventos, semilla = PRACTICA, 2026
        dir_est = dir_sol = a.salida
    else:
        if a.semilla is None:
            sys.exit('--evaluacion necesita --semilla (guardala: reproduce el caso)')
        semilla = a.semilla
        eventos = sortear_eventos(random.Random(semilla))
        dir_est = os.path.join(a.salida, 'estudiantes')
        dir_sol = os.path.join(a.salida, 'solucion_NO_DISTRIBUIR')
    os.makedirs(dir_est, exist_ok=True)
    os.makedirs(dir_sol, exist_ok=True)

    sim(f'{dir_est}/nominal_entrenamiento.csv', '--duration', 600, '--seed', semilla + 1)
    sim(f'{dir_est}/nominal_calibracion.csv', '--duration', 300, '--seed', semilla + 2)
    args = ['--duration', DUR, '--seed', semilla + 3, '--n-delays', 4]
    for tipo, ini, d in eventos:
        args += CATALOGO[tipo](ini, d)
    sim(f'{dir_est}/registro_auditoria.csv', *args)
    with open(f'{dir_est}/ficha_robot.txt', 'w', encoding='utf-8') as f:
        f.write(FICHA)

    nombre = 'eventos_practica.txt' if a.practica else 'eventos.txt'
    with open(f'{dir_sol}/{nombre}', 'w', encoding='utf-8') as f:
        f.write('# inicio fin   tipo   (s, relativos al primer mensaje)\n')
        for tipo, ini, d in eventos:
            f.write(f'{ini:g} {ini + d:g}   # {tipo}\n')
    with open(f'{dir_sol}/caso.json', 'w', encoding='utf-8') as f:
        json.dump({'semilla': semilla, 'duracion_s': DUR, 'retrasos_aislados': 4,
                   'eventos': [{'tipo': t, 'inicio': i, 'fin': i + d} for t, i, d in eventos]},
                  f, indent=2, ensure_ascii=False)
    print(f'Registros en {dir_est}/')
    if not a.practica:
        print(f'Solución en {dir_sol}/  (no distribuir)')
    for tipo, ini, d in eventos:
        print(f'  {tipo:11s} {ini:7.1f} - {ini + d:7.1f} s')


if __name__ == '__main__':
    main()
