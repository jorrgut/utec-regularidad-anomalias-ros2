#!/usr/bin/env bash
# Prueba rápida de los scripts del curso, sin ROS 2.
# Copia codigos/ a una carpeta temporal y recorre el flujo de las seis unidades
# y del proyecto integrador con datos sintéticos. No escribe en el repositorio.
#
# Uso:  bash tests/smoke_test.sh          (desde la raíz del repositorio)
#       SKIP_TORCH=1 bash tests/smoke_test.sh   (omite la Unidad 6)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp -r "$REPO/codigos" "$TMP/"
export MPLBACKEND=Agg
C="$TMP/codigos"
paso() { echo; echo "== $*"; }
q() { "$@" > "$TMP/ultimo.log" 2>&1 || { cat "$TMP/ultimo.log"; echo "FALLÓ: $*"; exit 1; }; }

paso "Sintaxis de todos los .py"
q python3 -m compileall -q "$C"

paso "Unidad 1: simulador y auditoría"
cd "$C/u1"
q python3 simulate_cmdvel_csv.py --out data/cmdvel_u1.csv --duration 300
q python3 quick_audit_dt.py --csv data/cmdvel_u1.csv --outdir outputs_u1

paso "Unidad 2: re-muestreo y ventanas"
cd "$C/u2"; mkdir -p data; cp ../u1/data/cmdvel_u1.csv data/
q python3 resample_cmdvel_20hz.py --csv data/cmdvel_u1.csv --method linear   --out data/cmdvel_20hz.csv
q python3 resample_cmdvel_20hz.py --csv data/cmdvel_u1.csv --method previous --out data/cmdvel_20hz_zoh.csv
q python3 window_features.py --csv data/cmdvel_u1.csv --mode samples --n 20 --stride 10 --out data/features_samples.csv
q python3 window_features.py --csv data/cmdvel_u1.csv --mode time --tw 2.0 --step 1.0 --out data/features_time.csv

paso "Unidad 3: entropía"
cd "$C/u3"; mkdir -p data
cp ../u2/data/cmdvel_20hz.csv ../u2/data/cmdvel_20hz_zoh.csv ../u1/data/cmdvel_u1.csv data/
q python3 entropy_windows.py --csv data/cmdvel_20hz.csv --compare data/cmdvel_20hz_zoh.csv \
    --labels linear ZOH --out outputs_u3a1/entropy_windows.csv
q python3 bins_sensitivity.py --csv data/cmdvel_20hz.csv --signal v --outdir outputs_u3a1
q python3 entropy_dt.py --csv data/cmdvel_u1.csv --miller-madow --out outputs_u3a1/entropy_dt.csv

paso "Unidad 4: predicción"
cd "$C/u4"; mkdir -p data; cp ../u2/data/cmdvel_20hz_zoh.csv data/
q python3 u4_activity1.py
q python3 u4_activity2.py
q python3 u4_activity3.py
q python3 predict_cmdvel.py --csv data/cmdvel_20hz_zoh.csv --out outputs_u4a4/zoh
q python3 simulate_cmdvel_csv.py --out data/replay.csv --replay-at 200 --replay-len 10 \
    --replay-shift 6.5 --replay-mode add
q python3 ../u2/resample_cmdvel_20hz.py --csv data/replay.csv --method previous --out data/replay_20hz_zoh.csv

paso "Unidad 5: calibración y evaluación (camino sin ROS)"
cd "$C/u5"
q python3 simulate_cmdvel_csv.py --out data/u5_base.csv --duration 600 --drop-prob 0.005 --quantum 0.001 --seed 1
q python3 simulate_cmdvel_csv.py --out data/u5_run.csv --duration 450 --drop-prob 0.005 --quantum 0.001 --seed 3 \
    --flood-at 120 --flood-len 20 --flood-rate 30 --replay-at 300 --replay-len 20 --replay-shift 6.5
for n in base run; do q python3 metrics_from_csv.py --csv data/u5_$n.csv --log-dir outputs_u5/sint_$n; done
printf "120 140  # flood\n300 320  # replay\n" > outputs_u5/sint_run/events.txt
q python3 calibrate_detector.py --log-dir outputs_u5/sint_base --p 0.99 --out outputs_u5/sint_calib.yaml
q python3 evaluate_detector.py --log-dir outputs_u5/sint_run --calib outputs_u5/sint_calib.yaml

paso "Proyecto integrador: caso de práctica"
cd "$C/integrador"
q python3 generar_caso.py --practica --salida caso_practica

if [ "${SKIP_TORCH:-0}" = "1" ]; then
  echo; echo "== Unidad 6 omitida (SKIP_TORCH=1)"
else
  paso "Unidad 6: datos y un autoencoder corto"
  cd "$C/u6"
  q python3 -c "import torch; print(torch.__version__)"
  q python3 generate_datasets.py
  q python3 train_model.py --kind ae --T 40 --out models/ae_T40
  q python3 u6_activity4.py --models models/ae_T40
fi

echo; echo "Todo OK."
