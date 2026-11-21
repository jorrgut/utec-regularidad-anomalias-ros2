#!/usr/bin/env python3
"""
Genera con el simulador todos los CSV de la Unidad 6 (sin ROS ni Gazebo).

    data/train_structured.csv   10 min de structured  (entrenamiento + validación por bloques)
    data/test_structured.csv     5 min de structured  (prueba nominal, otra corrida)
    data/shuffle_blocks.csv      5 min de shuffle_blocks
    data/iid.csv                 5 min de iid
    data/eventos.csv            10 min de structured con tres eventos:
                                   150-180 s  segmento shuffle_blocks (pérdida de estructura)
                                   300-310 s  flood a 40 Hz con valores legítimos
                                   450-460 s  replay por reemplazo (comandos de 6.5 s antes)
    data/eventos.txt            los intervalos de los eventos (formato de la Unidad 5)

Los tiempos de eventos.txt son relativos al primer mensaje, igual que los que usan los
scripts de esta unidad.

Uso:
    python3 generate_datasets.py            # en ./data
    python3 generate_datasets.py --out-dir data_otra_semilla --seed-offset 100
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, 'simulate_cmdvel_csv.py')


def run(out, *extra):
    cmd = [sys.executable, SIM, '--out', out, *map(str, extra)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(r.stderr)
    print('  ' + r.stdout.strip())


def main():
    ap = argparse.ArgumentParser(description='Datos sintéticos de la Unidad 6')
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--seed-offset', type=int, default=0)
    a = ap.parse_args()
    o, s = a.out_dir, a.seed_offset
    os.makedirs(o, exist_ok=True)
    run(f'{o}/train_structured.csv', '--duration', 600, '--seed', 1 + s)
    run(f'{o}/test_structured.csv', '--duration', 300, '--seed', 2 + s)
    run(f'{o}/shuffle_blocks.csv', '--duration', 300, '--mode', 'shuffle_blocks', '--seed', 3 + s)
    run(f'{o}/iid.csv', '--duration', 300, '--mode', 'iid', '--seed', 4 + s)
    run(f'{o}/eventos.csv', '--duration', 600, '--seed', 5 + s,
        '--segment-mode', 'shuffle_blocks', '--segment-at', 150, '--segment-len', 30,
        '--flood-at', 300, '--flood-len', 10, '--flood-rate', 40,
        '--replay-at', 450, '--replay-len', 10, '--replay-shift', 6.5, '--replay-mode', 'replace')
    with open(f'{o}/eventos.txt', 'w', encoding='utf-8') as f:
        f.write('# inicio fin   (s, relativos al primer mensaje)\n')
        f.write('150 180   # segmento shuffle_blocks\n')
        f.write('300 310   # flood 40 Hz, valores legítimos\n')
        f.write('450 460   # replay por reemplazo\n')
    print(f'  eventos en {o}/eventos.txt')


if __name__ == '__main__':
    main()
