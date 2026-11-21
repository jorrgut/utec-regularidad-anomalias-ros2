# Unidad 5: Detección de anomalías en tiempo real (score, umbral y persistencia)

**Guía práctica de laboratorio**
Optativa *Análisis de Regularidad y Detección de Anomalías en Sistemas Robóticos*, Ingeniería en Mecatrónica (UTEC)

Esta guía acompaña al apunte de la Unidad 5. Vas a construir un detector de anomalías completo: indicadores en línea, score, umbral calibrado y persistencia. Lo vas a operar en ROS 2 y a ajustarlo fuera de línea sobre registros. La teoría está en el apunte: normalización robusta, dirección del desvío, qué significa "1 % de falsas alarmas", por qué la persistencia rinde menos con scores correlacionados y cómo evaluar por eventos. Acá está el **cómo**.

---

## Índice

1. [Qué vas a hacer](#1-qué-vas-a-hacer)
2. [Los archivos de la unidad](#2-los-archivos-de-la-unidad)
3. [Instalación](#3-instalación)
4. [Camino A: practicar sin ROS (datos sintéticos)](#4-camino-a-practicar-sin-ros-datos-sintéticos)
5. [Camino B: el detector en ROS 2](#5-camino-b-el-detector-en-ros-2)
6. [Cómo leer los resultados](#6-cómo-leer-los-resultados)
7. [Las actividades, paso a paso](#7-las-actividades-paso-a-paso)
8. [Referencia: tópicos, CSV y parámetros](#8-referencia-tópicos-csv-y-parámetros)
9. [Entregables y checklist](#9-entregables-y-checklist)
10. [Resolución de problemas](#10-resolución-de-problemas)
11. [Referencia rápida](#11-referencia-rápida)

---

## 1. Qué vas a hacer

```
                       EN LÍNEA (ROS 2)                               FUERA DE LÍNEA
 cmd_vel_generator ─┐
 cmdvel_attacker  ──┴─► /cmd_vel ─┬─► entropy_metrics  ─► /entropy_metrics ─┐
                                  └─► cmdvel_predictor ─► /cmd_vel_pred_metrics ─┤
                                                                            ▼
                         anomaly_detector ◄── calib.yaml ◄── calibrate_detector.py
                         (score, τ, K)                              ▲
                              │                                     │
                              ▼                                     │
               /anomaly_score, /anomaly_alert, /anomaly_debug       │
                              │                                     │
                     csv_logger ──► CSV ────────────────────────────┴──► evaluate_detector.py
                                                                         (grilla τ × K, CUSUM)
```

Los indicadores son los de las unidades anteriores, calculados en línea sobre ventanas de 5 s:

| Indicador | Qué mide | Unidad de origen | Dirección |
|---|---|---|---|
| `mse_v`, `mse_w` | Error de persistencia de v y ω | 4 | `up` (solo cuenta subir) |
| `Hcond_v` | Entropía condicional H(v_t+1 \| v_t): desorden en el orden | 3 | `up` |
| `H_dt` | Entropía de los intervalos Δt: ritmo de llegada | 3 | `both` (cualquier desvío) |
| `H_v` | Entropía de Shannon de v: forma de la distribución | 3 | `both` |

Al final vas a poder responder, con tus propios datos:

- ¿Cuántas falsas alarmas **por hora** da tu detector en operación normal?
- ¿Qué anomalías detecta, con qué retardo, y **qué indicador** causó cada alarma?
- ¿Qué configuración de τ y K elegirías, y por qué?

**Tiempo estimado de laboratorio:** unas 6 h.

---

## 2. Los archivos de la unidad

```
codigos_u5/
├── u5_core.py              ← núcleo de cálculo: lo usan los nodos Y los scripts
├── u5_offline.py           ← lectura de CSV y estilo de figuras (scripts fuera de línea)
├── metrics_from_csv.py     ← CSV de /cmd_vel → los mismos registros que los nodos de métricas
├── calibrate_detector.py   ← línea base → calib.yaml
├── evaluate_detector.py    ← registro + calib.yaml → evaluación por eventos, grillas, figuras
├── simulate_cmdvel_csv.py  ← el simulador de las Unidades 3 y 4 (flood, replay)
└── cmdvel_tools_nodes/     ← nodos ROS 2 (van dentro del paquete cmdvel_tools)
    ├── u5_core.py          ← la MISMA copia del núcleo
    ├── cmd_vel_generator.py
    ├── cmdvel_attacker.py
    ├── entropy_metrics.py
    ├── cmdvel_predictor.py
    ├── anomaly_detector.py
    ├── csv_logger.py
    └── setup_entry_points.txt
```

> **Por qué un núcleo compartido:** los nodos y los scripts calculan los indicadores, el score y la persistencia con **el mismo código**. Lo que ajustes fuera de línea es exactamente lo que hace el detector en línea.

---

## 3. Instalación

### 3.1 Scripts fuera de línea (no requieren ROS)

No hay que copiarlos: están en `~/project/codigos/u5` y se ejecutan desde esa carpeta (importan `u5_core.py` y `u5_offline.py`).

```bash
cd ~/project/codigos/u5
python3 calibrate_detector.py --help     # verificación
```

Requieren `numpy`, `pandas`, `matplotlib` y `pyyaml`. Los cuatro ya están en el contenedor; PyYAML viene con ROS 2.

### 3.2 Nodos ROS 2 (dentro del contenedor)

1. Copiá **todo** el contenido de `cmdvel_tools_nodes/` (incluidos `u5_core.py` y `cmdvel_msg.py`) a la carpeta de código del paquete de la Unidad 1:
   ```bash
   cd ~/project/codigos/u5
   cp cmdvel_tools_nodes/*.py ~/project/ros2_ws/src/cmdvel_tools/cmdvel_tools/
   ```
2. Agregá los ejecutables a `~/project/ros2_ws/src/cmdvel_tools/setup.py`. Las líneas exactas están en `cmdvel_tools_nodes/setup_entry_points.txt`.
3. Compilá y activá:
   ```bash
   cd ~/project/ros2_ws
   colcon build
   source install/setup.bash
   ros2 pkg executables cmdvel_tools
   ```
   Tenés que ver los seis nodos (más `dummy_node`): `anomaly_detector`, `cmd_vel_generator`, `cmdvel_attacker`, `cmdvel_predictor`, `csv_logger` y `entropy_metrics`.

> Si después modificás `u5_core.py`, copialo **a los dos lugares** y volvé a compilar.

---

## 4. Camino A: practicar sin ROS (datos sintéticos)

Antes de operar el robot, recorré el ciclo completo con datos sintéticos. Todo corre desde `~/project/codigos/u5` y tarda pocos segundos.

```bash
cd ~/project/codigos/u5

# 1. Tres registros: línea base (10 min), validación nominal (5 min), corrida con ataques
python3 simulate_cmdvel_csv.py --out data/u5_base.csv --duration 600 --drop-prob 0.005 \
    --quantum 0.001 --seed 1
python3 simulate_cmdvel_csv.py --out data/u5_val.csv --duration 300 --drop-prob 0.005 \
    --quantum 0.001 --seed 2
python3 simulate_cmdvel_csv.py --out data/u5_run.csv --duration 450 --drop-prob 0.005 \
    --quantum 0.001 --seed 3 --flood-at 120 --flood-len 20 --flood-rate 30 \
    --replay-at 300 --replay-len 20 --replay-shift 6.5

# 2. Indicadores "en línea" (lo mismo que publicarían los nodos)
for n in base val run; do
  python3 metrics_from_csv.py --csv data/u5_$n.csv --log-dir outputs_u5/sint_$n
done
printf "120 140  # flood\n300 320  # replay\n" > outputs_u5/sint_run/events.txt

# 3. Calibrar con la línea base
python3 calibrate_detector.py --log-dir outputs_u5/sint_base --p 0.99 --out outputs_u5/sint_calib.yaml

# 4. Validar (falsas alarmas) y evaluar la corrida con ataques
python3 evaluate_detector.py --log-dir outputs_u5/sint_val --calib outputs_u5/sint_calib.yaml
python3 evaluate_detector.py --log-dir outputs_u5/sint_run --calib outputs_u5/sint_calib.yaml
```

**Lo que deberías obtener** (las semillas son fijas, así que los números se repiten exactamente):

| Paso | Resultado |
|---|---|
| Calibración | τ = Q0.99 ≈ 4.57. Aviso: unas 190 muestras "efectivas" de 1179, porque el score tiene correlación 0.72 entre registros consecutivos. `mse_v`, `Hcond_v` y `H_v` muy correlacionados (\|r\| > 0.7). |
| Validación (τ = Q0.99, K = 3) | 1 falsa alarma en 5 min (≈ 12 por hora), causada por `H_dt` |
| Corrida con ataques | 2 de 2 eventos detectados, retardo medio ≈ 1.5 s, 0 falsas alarmas. El flooding lo detecta `H_dt` y el replay, `Hcond_v`. |

---

## 5. Camino B: el detector en ROS 2

Necesitás **siete terminales** dentro del contenedor (`docker exec -it utec_ros2_humble bash` en cada una), con el workspace activado y `cd ~/project/codigos/u5` (así los registros quedan en `outputs_u5/` de esa carpeta).

### 5.1 Línea base (Actividad 1)

```bash
# T1: simulación
ros2 launch turtlebot3_gazebo empty_world.launch.py

# T2: generador nominal
ros2 run cmdvel_tools cmd_vel_generator --ros-args -p use_sim_time:=true -p mode:=structured

# T3 y T4: métricas
ros2 run cmdvel_tools entropy_metrics  --ros-args -p use_sim_time:=true
ros2 run cmdvel_tools cmdvel_predictor --ros-args -p use_sim_time:=true

# T5: logger
ros2 run cmdvel_tools csv_logger --ros-args -p use_sim_time:=true -p log_dir:=outputs_u5/baseline
```

Dejalo correr **al menos 10 minutos de tiempo simulado**, y cortá el logger con Ctrl+C. Después repetí con `log_dir:=outputs_u5/validation` durante 5 minutos más.

```bash
python3 calibrate_detector.py --log-dir outputs_u5/baseline --p 0.99 --out outputs_u5/calib.yaml
python3 evaluate_detector.py  --log-dir outputs_u5/validation --calib outputs_u5/calib.yaml
```

### 5.2 Corrida con el detector y ataques (Actividad 2)

Con T1 a T4 corriendo:

```bash
# T5: logger de la corrida
ros2 run cmdvel_tools csv_logger --ros-args -p use_sim_time:=true -p log_dir:=outputs_u5/run1

# T6: detector
ros2 run cmdvel_tools anomaly_detector --ros-args -p use_sim_time:=true \
    -p calib_file:=outputs_u5/calib.yaml -p K:=3

# T7: ataques (uno por vez; cada uno termina solo)
ros2 run cmdvel_tools cmdvel_attacker --ros-args -p use_sim_time:=true \
    -p mode:=flood -p start_after:=30.0 -p duration:=20.0 \
    -p events_file:=outputs_u5/run1/events.txt
ros2 run cmdvel_tools cmdvel_attacker --ros-args -p use_sim_time:=true \
    -p mode:=replay -p start_after:=30.0 -p duration:=20.0 \
    -p events_file:=outputs_u5/run1/events.txt
```

El atacante imprime el intervalo del ataque y lo **agrega** a `events.txt`, en el mismo reloj que los CSV. Para observar el detector en vivo, en otra terminal:

```bash
ros2 topic echo /anomaly_alert
ros2 topic echo /anomaly_debug     # [t, S, tau, alarma, contribuciones...]
```

**Pérdida de estructura:** cortá el generador (Ctrl+C en T2) y relanzalo con `-p mode:=shuffle_blocks -p block_len:=30`, y después con `-p mode:=iid`. Anotá los instantes a mano en `events.txt`; el formato es una línea `inicio fin` por evento, en sim time, que podés leer en la columna `t` de los CSV o en `ros2 topic echo /clock`.

Al terminar:

```bash
python3 evaluate_detector.py --log-dir outputs_u5/run1 --calib outputs_u5/calib.yaml
```

---

## 6. Cómo leer los resultados

### `calibrate_detector.py`

```
  indicador     centro     escala   fuente  peso  dir.   NaN
  mse_v         0.0008  2.372e-05      mad  0.50    up     0
  ...
tau = cuantil 0.99 = 4.568
Correlación del score entre registros consecutivos: 0.72 -> unas 189 muestras "efectivas" de 1179
  [aviso] con ~189 muestras efectivas, el cuantil 0.99 es poco confiable. ...
Indicadores muy correlacionados en la línea base (|r| > 0.7): mse_v-Hcond_v (+0.86), ...
```

| Lo que ves | Qué significa |
|---|---|
| `fuente` = `mad` | Escala robusta normal: 1.4826·MAD |
| `fuente` = `std` o `piso` | La MAD dio 0: el indicador toma muy pocos valores en la línea base (apunte, "normalizar con una MAD nula"). Su z-score merece una mirada. |
| Escala muy chica frente al centro (como `mse_v`) | El indicador casi no varía en operación normal: cualquier cambio da z enormes. La **saturación** (`--z-max`, 10 por defecto) evita que domine el score. |
| Muestras "efectivas" | Cuánta información independiente tiene la línea base. Si son pocas, el cuantil alto es incierto: registrá más tiempo o usá un `--p` menor. |
| Indicadores correlacionados | Están contando la misma evidencia. Considerá bajar el peso de alguno: `--weights H_v=0.5`. |

### `evaluate_detector.py`

```
Configuración principal (tau = 4.568, K = 3):
  eventos detectados: 2 de 2 | retardo medio 1.50 s
  falsas alarmas: 0 (0.0 por hora) | tiempo nominal en alarma: 0.0%
    evento 120-140 s: primera alarma en 121.5 s (indicador dominante: H_dt)
    evento 300-320 s: primera alarma en 301.5 s (indicador dominante: Hcond_v)

Grilla tau x K (falsas alarmas por hora ...):
K           1      3      5      10
cuantil
0.900    240.5   96.2   40.1    8.0
...
```

| Línea o tabla | Qué significa |
|---|---|
| eventos detectados / retardo | Evaluación **por eventos** (apunte, "Evaluar un detector") |
| falsas alarmas por hora | Intervalos de alarma que no se superponen con ningún evento ni con su margen (5 s) |
| tiempo nominal en alarma | Detecta el caso patológico de una alarma que nunca se apaga: cuenta como **una sola** "falsa alarma" pero tiene el robot detenido todo el tiempo |
| indicador dominante | El término que más aportó al score cuando empezó la alarma. Compará con la tabla de la sección 1. |
| Grilla τ × K | Cada celda es un detector posible. Bajar el cuantil o K: más sensible, más falsas alarmas. Subir: menos falsas alarmas, más retardo. |
| CUSUM | Trabaja sobre el score **centrado** en su mediana nominal y acotado (apunte, sección del CUSUM). `k` es la "holgura" y `h` el umbral del acumulado. |

### Figuras (en `<log-dir>/eval/` y junto a la calibración)

| Figura | Qué mirar |
|---|---|
| `baseline_score.png` | El score nominal con τ: ¿hay rachas sobre τ? ¿Cuánto duran? |
| `run_score.png` | Arriba, los z de cada indicador; al medio, sus contribuciones apiladas (de qué está hecho el score); abajo, el score con τ y las alarmas. Los eventos están sombreados. |
| `tradeoff.png` | Falsas alarmas por hora contra retardo medio: una curva por K y un punto por umbral |

---

## 7. Las actividades, paso a paso

### Actividad 1: calibración nominal
1. Línea base de ≥ 10 min y validación de ≥ 5 min (sección 5.1).
2. Calibrá con `--p 0.99` y con `--p 0.95`, con `--out` distintos.
3. Evaluá ambas sobre la **validación**. Compará las falsas alarmas por hora con lo que "prometía" p.
4. Mirá `baseline_correlation.csv` y decidí si ajustás pesos.

### Actividad 2: detector en línea
1. Corrida con flood, replay y cambios de modo (sección 5.2).
2. Mirá `/anomaly_debug` en vivo y la figura `run_score.png` después.
3. Para cada alarma: ¿qué indicador dominó? ¿Coincide con la tabla de la sección 1?
4. Al reiniciar el generador para cambiar de modo hay unos segundos sin mensajes. ¿Generan alarma? ¿Es "falsa"?

### Actividad 3: compromiso entre τ, K y retardo
1. `evaluate_detector.py` sobre la corrida de la Actividad 2: grillas de τ × K y CUSUM.
2. Compará la reducción de falsas alarmas al pasar de K = 1 a K = 3 con la predicción ingenua (1 − p)^K.
3. Elegí una configuración, cargala en el nodo (`-p K:=...`, y `-p tau:=...` si no usás el de la calibración, o `-p rule:=cusum`) y verificala en una corrida nueva.

### Actividad 4: discusión comparativa
Sin código nuevo. Usá tus resultados y la tabla de la sección 1. Una pista para la pregunta del replay por reemplazo: el simulador lo genera con `--replay-mode replace`. Probá cómo responde tu detector con el camino A.

---

## 8. Referencia: tópicos, CSV y parámetros

### Tópicos

| Tópico | Tipo | Contenido |
|---|---|---|
| `/entropy_metrics` | `std_msgs/Float64MultiArray` | `[t, n_msgs, H_v, Hcond_v, H_dt]` |
| `/cmd_vel_pred_metrics` | `std_msgs/Float64MultiArray` | `[t, n_msgs, mse_v, mse_w]` |
| `/anomaly_score` | `std_msgs/Float64` | score S_t |
| `/anomaly_alert` | `std_msgs/Bool` | alarma A_t |
| `/anomaly_debug` | `std_msgs/Float64MultiArray` | `[t, S, tau, alarma, c_mse_v, c_mse_w, c_Hcond_v, c_H_dt, c_H_v]` |

Un indicador vale `NaN` si la ventana tiene menos de `min_samples` mensajes, por ejemplo durante el calentamiento o una pausa. En el score, un NaN aporta 0: no hay evidencia.

### CSV que escribe `csv_logger` (en `log_dir`)

`cmd_vel.csv` (el formato de la Unidad 1, reutilizable con `metrics_from_csv.py`), `entropy_metrics.csv`, `cmd_vel_pred_metrics.csv` y `anomaly_debug.csv`, con las columnas de los tópicos. Los tiempos son **sim time**.

### Parámetros de los nodos

| Nodo | Parámetro | Por defecto | Significado |
|---|---|---|---|
| `cmd_vel_generator` | `mode` | `structured` | `structured`, `shuffle_blocks` o `iid` |
| | `rate_hz`, `v_fwd`, `w_turn`, `t_fwd`, `t_turn`, `t_stop` | 10, 0.20, 0.80, 6, 3, 2 | Como el generador de la Unidad 1 |
| | `block_len` | 30 | Mensajes por bloque en `shuffle_blocks` |
| | `seed`, `duration` | 0, 0 | Semilla; duración (0 = sin límite) |
| `cmdvel_attacker` | `mode` | `flood` | `flood` o `replay` |
| | `start_after`, `duration` | 60, 20 | Segundos desde que arranca el nodo; duración del ataque |
| | `rate_hz` | 30 | Mensajes extra por segundo (flood) |
| | `replay_shift`, `replay_rate_hz` | 6.5, 10 | Antigüedad de lo reenviado (s); ritmo del replay |
| | `events_file` | (vacío) | Archivo al que agrega `inicio fin` |
| `entropy_metrics` | `window_sec`, `min_samples`, `publish_hz` | 5, 30, 2 | Ventana por tiempo, mínimo de mensajes, frecuencia de publicación |
| | `v_min`, `v_max`, `n_bins`, `dt_max`, `n_bins_dt` | −0.22, 0.22, 15, 0.3, 20 | Grillas fijas (Unidad 3) |
| `cmdvel_predictor` | `window_sec`, `min_samples`, `publish_hz` | 5, 30, 2 | Igual que arriba |
| `anomaly_detector` | `calib_file` | (obligatorio) | YAML de `calibrate_detector.py` |
| | `tau` | 0 | Umbral; ≤ 0 usa el de la calibración |
| | `K` | 3 | Superaciones consecutivas |
| | `rule`, `cusum_k`, `cusum_h` | `consecutive`, 2, 5 | Regla de persistencia; parámetros del CUSUM |
| | `min_ready_sec`, `max_age_sec` | 10, 2 | Calentamiento; antigüedad máxima de un indicador |
| todos los que usan `/cmd_vel` | `stamped` | según `ROS_DISTRO` | `Twist` (Humble) o `TwistStamped` (Jazzy en adelante); lo resuelve `cmdvel_msg.py`. Con `TwistStamped`, `csv_logger` agrega la columna `t_stamp` |
| `csv_logger` | `log_dir` | `outputs_u5/run` | Carpeta de salida |

Todos los nodos aceptan `-p use_sim_time:=true`, y **hay que usarlo en todos**.

---

## 9. Entregables y checklist

| Actividad | Entregar | Informe |
|---|---|---|
| 1 | `calib.yaml` (las dos versiones de p), `baseline_score.png`, evaluación de la validación | 8–12 líneas |
| 2 | CSV de la corrida, `events.txt`, `run_score.png` | 10–15 líneas |
| 3 | `grid_consecutive.csv`, `grid_cusum.csv`, `tradeoff.png`, configuración elegida y su verificación | 12–18 líneas |
| 4 | — | 15–20 líneas |

### ✅ Checklist antes de entregar

- [ ] La línea base dura ≥ 10 min y la validación es **otra** corrida
- [ ] Reporto falsas alarmas **por hora**, medidas en la validación
- [ ] Revisé la fuente de la escala de cada indicador y las correlaciones
- [ ] Evalué por eventos: detectados, retardo y falsas alarmas por hora
- [ ] Atribuí cada alarma a un indicador
- [ ] Comparé la reducción real por K con (1 − p)^K y expliqué la diferencia
- [ ] Verifiqué en línea la configuración elegida fuera de línea
- [ ] Discutí causas no maliciosas y lo que el detector no ve
- [ ] Las figuras tienen ejes, unidades y leyendas

### Qué se evalúa

| Criterio | Insuficiente | Aceptable | Destacado |
|---|---|---|---|
| Comprensión conceptual | Confunde muestras sobre el umbral con falsas alarmas, o suma z-scores con signo sin advertir cancelaciones | Explica el esquema score–umbral–persistencia y el compromiso entre falsas alarmas y retardo | Explica por qué la persistencia rinde menos con scores correlacionados y qué anomalías no puede ver el detector |
| Justificación técnica | Elige τ, K y pesos sin argumento, o calibra y evalúa sobre los mismos datos | Calibra con datos nominales y justifica con reglas generales | Justifica con evidencia propia: validación separada, correlación entre indicadores, curva de falsas alarmas contra retardo |
| Análisis crítico | Reporta alarmas sin compararlas con los eventos conocidos | Evalúa por eventos (detectados, retardo, falsas alarmas por hora) | Atribuye cada alarma a un indicador, discute causas no maliciosas y propone qué información adicional haría falta |

---

## 10. Resolución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ros2 run ...: No executable found` | Faltan los `entry_points` o no recompilaste | Revisá `setup.py` (sección 3.2), `colcon build`, `source install/setup.bash` |
| `ModuleNotFoundError: cmdvel_tools.u5_core` | Falta `u5_core.py` dentro del paquete | Copialo a `~/project/ros2_ws/src/cmdvel_tools/cmdvel_tools/` y recompilá |
| `ModuleNotFoundError: cmdvel_tools.cmdvel_msg` | Falta `cmdvel_msg.py` dentro del paquete | Copiá todos los `.py` de `cmdvel_tools_nodes/` y recompilá |
| En Jazzy el robot no se mueve o `/cmd_vel` tiene dos tipos | Algún nodo usa `Twist` y TurtleBot3 espera `TwistStamped` | Verificá `echo $ROS_DISTRO`; forzá `-p stamped:=true` |
| `ModuleNotFoundError: yaml` | Falta PyYAML (fuera del contenedor) | `pip install pyyaml` |
| `anomaly_detector`: `falta el parámetro calib_file` | No pasaste la calibración | `-p calib_file:=outputs_u5/calib.yaml` (calibrá antes) |
| Los nodos no publican nada | Con `use_sim_time`, `/clock` no avanza | `ros2 topic hz /clock`: Gazebo tiene que estar corriendo |
| Indicadores `NaN` todo el tiempo | Menos de `min_samples` mensajes por ventana | ¿Corre el generador? `ros2 topic hz /cmd_vel` |
| `calibrate_detector.py`: `línea base demasiado corta` | Menos de 50 registros tras el calentamiento | Registrá más tiempo |
| Muchas falsas alarmas en la validación | τ estimado con poca línea base, o la validación difiere de la línea base | Línea base más larga, `--p` menor, o K mayor; revisá si cambió algo |
| La alarma no se apaga nunca | Con `rule:=cusum`: calibración vieja sin `score_median` | Recalibrá con esta versión de `calibrate_detector.py` |
| `evaluate_detector.py` no ve los eventos | `events.txt` en otro lugar, o tiempos en otro reloj | Los eventos van en **sim time**, como la columna `t` de los CSV |
| Alarma al cambiar de modo el generador | Unos segundos sin mensajes: `H_dt` y `n_msgs` cambian | Es esperado. Decidí en el informe si es "falsa" o un evento real. |
| Cambié `u5_core.py` y los nodos no reflejan el cambio | La copia del paquete es otra | Copialo a los dos lugares y recompilá |

---

## 11. Referencia rápida

```bash
# Fuera de línea
python3 metrics_from_csv.py   --csv data/cmdvel_u1.csv --log-dir outputs_u5/x
python3 calibrate_detector.py --log-dir outputs_u5/baseline --p 0.99 --out outputs_u5/calib.yaml
python3 calibrate_detector.py --log-dir outputs_u5/baseline --weights H_v=0.5 --z-max 10
python3 evaluate_detector.py  --log-dir outputs_u5/run1 --calib outputs_u5/calib.yaml
python3 evaluate_detector.py  --log-dir outputs_u5/run1 --calib outputs_u5/calib.yaml \
    --events 120 140 300 320 --K 5 --K-grid 1 3 5 10

# En línea
ros2 run cmdvel_tools cmd_vel_generator --ros-args -p use_sim_time:=true -p mode:=structured
ros2 run cmdvel_tools entropy_metrics   --ros-args -p use_sim_time:=true
ros2 run cmdvel_tools cmdvel_predictor  --ros-args -p use_sim_time:=true
ros2 run cmdvel_tools anomaly_detector  --ros-args -p use_sim_time:=true \
    -p calib_file:=outputs_u5/calib.yaml -p K:=3          # o -p rule:=cusum
ros2 run cmdvel_tools csv_logger        --ros-args -p use_sim_time:=true -p log_dir:=outputs_u5/run1
ros2 run cmdvel_tools cmdvel_attacker   --ros-args -p use_sim_time:=true -p mode:=flood \
    -p start_after:=30.0 -p events_file:=outputs_u5/run1/events.txt
```

---

### Lecturas para profundizar

- Montgomery (2019), *Introduction to Statistical Quality Control*, capítulos sobre gráficos de control CUSUM.
- Lavin y Ahmad (2015), *Evaluating real-time anomaly detection algorithms*: cómo evaluar detectores en tiempo real de forma justa.
- La bibliografía completa, organizada por tema, está al final del apunte de la unidad.
