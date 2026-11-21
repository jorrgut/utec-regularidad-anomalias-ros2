# Unidad 6: Modelos aprendidos para detección de anomalías (autoencoders y LSTM)

**Guía práctica de laboratorio**
Optativa *Análisis de Regularidad y Detección de Anomalías en Sistemas Robóticos*, Ingeniería en Mecatrónica (UTEC)

Esta guía acompaña al apunte de la Unidad 6. Vas a entrenar autoencoders y redes LSTM con datos nominales de `/cmd_vel`, medir qué ven y qué no, y usar su error como un indicador más del detector de la Unidad 5. La teoría está en el apunte: por qué un modelo puntual equivale a mirar la distribución (como la entropía de Shannon), por qué la ventana tiene que cubrir la escala del patrón, cómo partir los datos sin fuga y qué ataques quedan invisibles. Acá está el **cómo**.

---

## Índice

1. [Qué vas a hacer](#1-qué-vas-a-hacer)
2. [Los archivos de la unidad](#2-los-archivos-de-la-unidad)
3. [Instalación](#3-instalación)
4. [Los datos](#4-los-datos)
5. [Las actividades, paso a paso](#5-las-actividades-paso-a-paso)
6. [El modelo en línea en ROS 2 (opcional)](#6-el-modelo-en-línea-en-ros-2-opcional)
7. [Usar tus propios registros](#7-usar-tus-propios-registros)
8. [Referencia: archivos, columnas y parámetros](#8-referencia-archivos-columnas-y-parámetros)
9. [Entregables y checklist](#9-entregables-y-checklist)
10. [Resolución de problemas](#10-resolución-de-problemas)
11. [Referencia rápida](#11-referencia-rápida)

---

## 1. Qué vas a hacer

```
 CSV de /cmd_vel ─► grilla 10 Hz (ZOH) ─► estandarizar ─► ventanas de T muestras
 (logger o            + canal n_k            (stats de      │
  simulador)          (mensajes/celda)        train)        ▼
                                                    ┌──────────────────┐
                                                    │ AE puntual (T=1) │  actividad 1
                                                    │ AE de ventana    │  actividad 2
                                                    │ LSTM predictor   │  actividad 3
                                                    └────────┬─────────┘
                                                             │ error e_t
                                                             ▼
                indicadores de la Unidad 5 ─► z robusto ─► score S_t ─► τ, K ─► alarma
                                                                         │
                                                         evaluación por eventos (actividad 4)
```

Al final vas a poder responder, con tus propios datos:

- ¿Por qué el autoencoder puntual no distingue `shuffle_blocks` de `structured`?
- ¿Qué longitud de ventana T hace falta, y por qué?
- ¿La LSTM aprende algo más que "repetir el último valor"? ¿Le gana a un AR lineal?
- ¿Qué agrega el error del modelo al detector de la Unidad 5, y qué ataques siguen invisibles?

**Tiempo estimado de laboratorio:** unas 7 h (la actividad 3 es optativa).

---

## 2. Los archivos de la unidad

| Archivo | Para qué sirve |
|---|---|
| `simulate_cmdvel_csv.py` | Simulador de la Unidad 5 con dos agregados: `--mode` (structured, shuffle_blocks, iid) y un segmento anómalo (`--segment-mode/--segment-at/--segment-len`) |
| `generate_datasets.py` | Genera con el simulador todos los CSV de la unidad y `eventos.txt` |
| `u6_data.py` | Lectura de CSV, re-muestreo ZOH con canal `n_k`, ventanas, partición por bloques, estandarización, H(v) |
| `u6_models.py` | Modelos en PyTorch (`WindowAE`, `LSTMPredictor`), entrenamiento con parada temprana, errores, PCA, guardar y cargar |
| `u5_core.py` | Copia del núcleo de la Unidad 5 (indicadores, score, umbral, persistencia, evaluación por eventos) |
| `u6_activity1.py` | Actividad 1: autoencoder puntual como control |
| `u6_activity2.py` | Actividad 2: autoencoder de ventana, barrido de T y L, PCA |
| `u6_activity3.py` | Actividad 3: LSTM frente a persistencia y AR(p) |
| `train_model.py` | Entrena **un** modelo y lo guarda con todo lo necesario para usarlo |
| `u6_activity4.py` | Actividad 4: el error del modelo en el detector de la Unidad 5, evaluado por eventos |
| `cmdvel_tools_nodes/model_error.py` | Nodo de ROS 2 que calcula en línea el error de un modelo guardado |
| `cmdvel_tools_nodes/csv_logger.py` | El registrador de la Unidad 5 con un tópico más: `/model_metrics` |
| `cmdvel_tools_nodes/setup_entry_points.txt` | Qué agregar a `setup.py` para el nodo |

Las actividades escriben en `outputs_u6/actN/`; los modelos de `train_model.py`, en `models/`.

---

## 3. Instalación

Los scripts están en `~/project/codigos/u6` y se ejecutan **desde esa carpeta** (importan `u6_data.py`, `u6_models.py` y `u5_core.py`, que están junto a ellos).

Además de `numpy`, `pandas` y `matplotlib`, esta unidad necesita **PyTorch**. La imagen del curso ya lo trae (2.5.1, versión para CPU). Verificalo:

```bash
cd ~/project/codigos/u6
python3 -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"
# esperado: torch 2.5.1+cpu | numpy 1.26.4
```

- **Si construiste la imagen sin PyTorch** (`INSTALL_TORCH=0`): `pip3 install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu`. No actualices numpy: ROS 2 Humble necesita numpy 1.x.
- **Fuera del contenedor:** las actividades 1 a 4 son fuera de línea y no necesitan ROS 2. Podés correrlas en un entorno virtual (`python3 -m venv venv && source venv/bin/activate`, y después `pip install "numpy<2" pandas matplotlib pyyaml` más el comando de PyTorch).
- **No hace falta GPU.** Las redes tienen entre 3 000 y 21 000 parámetros.

Si falta PyTorch, los scripts lo avisan con el comando de instalación en lugar de mostrar un error largo.

---

## 4. Los datos

```bash
cd ~/project/codigos/u6
python3 generate_datasets.py
```

genera en `data/`:

| Archivo | Contenido | Uso |
|---|---|---|
| `train_structured.csv` | 10 min de `structured` | entrenar (primer 80 %) y validar (último 20 %) |
| `test_structured.csv` | 5 min de `structured`, otra corrida | prueba nominal; calibración del score en la actividad 4 |
| `shuffle_blocks.csv` | 5 min de `shuffle_blocks` | mismos valores, sin la estructura temporal |
| `iid.csv` | 5 min de `iid` | valores nuevos, sin estructura |
| `eventos.csv` | 10 min de `structured` con tres eventos | evaluación por eventos (actividad 4) |
| `eventos.txt` | los intervalos de los eventos | formato de la Unidad 5 |

Los eventos de `eventos.csv` son:

| Intervalo (s) | Evento | Qué cambia |
|---|---|---|
| 150–180 | segmento `shuffle_blocks` | el orden y la duración de las fases |
| 300–310 | flood a 40 Hz con valores legítimos | solo el ritmo de llegada |
| 450–460 | replay por reemplazo (comandos de 6.5 s antes) | nada, salvo en las "costuras" |

Todos los scripts leen el CSV **original** (el del logger: `t_sec, v, w`) y re-muestrean por su cuenta a una grilla de 10 Hz con retención de orden cero (*previous*), huecos en NaN (más de 0.5 s sin mensajes) y el canal `n_k` (cantidad de mensajes en cada celda de 100 ms). No le pases un CSV ya re-muestreado: se perdería `n_k`.

> **Detalle de la grilla.** Los bordes de las celdas se ubican a medio período de la fase típica de llegada de los mensajes. Si un borde coincidiera con los instantes de llegada, el jitter haría que algunas celdas tengan 0 mensajes y la siguiente 2, y `n_k` sería ruidoso aun en operación nominal.

---

## 5. Las actividades, paso a paso

Los números de esta sección salen de los datos de `generate_datasets.py` con las semillas por defecto. Los tuyos pueden diferir en los decimales (dependen de la versión de PyTorch y del procesador), pero las conclusiones no deberían cambiar.

### Actividad 1: el autoencoder puntual como control (≈ 40 min)

```bash
python3 u6_activity1.py
```

Entrena un autoencoder 2 → 16 → 1 → 16 → 2 sobre los puntos (v, ω) estandarizados de `structured` (es el modelo que proponía la versión anterior de la práctica) y lo evalúa en los tres modos.

**Qué deberías ver** (`outputs_u6/act1/act1_resumen.csv`):

| Modo | Mediana del error | Percentil 99 | H(v) medio (bits) |
|---|---|---|---|
| structured | 1.1 × 10⁻¹⁰ | 1.5 × 10⁻¹⁰ | 0.55 |
| shuffle_blocks | 1.1 × 10⁻¹⁰ | 1.5 × 10⁻¹⁰ | 0.40 |
| iid | 0.34 | 3.6 | 2.82 |

El error en `shuffle_blocks` es **el mismo** que en `structured`: los dos usan los mismos tres pares (v, ω), y el error de cada par es el mismo en cualquier orden (el script también lo imprime par por par). El valor de 10⁻¹⁰ es "cero" en precisión simple. `iid` se separa por ocho órdenes de magnitud porque aparecen valores que el modelo nunca vio. Es exactamente lo que hace H(v): ve valores nuevos, no el orden.

En `act1_error.png`, la escala es logarítmica (se suma 10⁻¹² para poder graficar el cero).

**Preguntas:** ¿por qué H(v) es algo distinta entre `structured` y `shuffle_blocks` si el error es igual? (Pista: proporción del tiempo en cada fase.) ¿Qué tendría que cambiar en la entrada para que el modelo distinga los dos modos?

### Actividad 2: autoencoder de ventana y elección de T (≈ 2 h)

```bash
python3 u6_activity2.py                         # completa: unos 4 min en una CPU de laptop
python3 u6_activity2.py --T 10 40 --seeds 0 --latent-sweep   # versión rápida
```

Para T ∈ {10, 40, 80} y tres semillas entrena un `WindowAE` (2T → 64 → 4 → 64 → 2T), fija τ en el cuantil 0.99 del error de validación, y mide qué fracción de ventanas de cada modo supera τ. Repite con PCA (el autoencoder lineal óptimo, misma L) y, para T = 40, barre el latente L ∈ {2, 4, 16}.

**Qué deberías ver** (fracción de ventanas con error > τ, `act2_resumen.csv`):

| Modelo | T | L | structured | shuffle_blocks | iid |
|---|---|---|---|---|---|
| AE | 10 | 4 | 0.01 | 0.05 | 1.00 |
| AE | 40 | 4 | 0.01 | **0.76** | 1.00 |
| AE | 80 | 4 | 0.01 | **1.00** | 1.00 |
| PCA | 10 | 4 | 0.01 | 0.05 | 0.85 |
| PCA | 40 | 4 | 0.01 | 0.66 | 1.00 |
| PCA | 80 | 4 | 0.01 | 0.97 | 1.00 |
| AE | 40 | 2 / 16 | 0.01 | 0.76 / 0.76 | 1.00 |
| PCA | 40 | 2 / 16 | 0.01 | 0.44 / 0.20 | 0.69 / 1.00 |

Cómo leerla:

- **En nominal, 0.01 en todos los casos**: el umbral está bien calibrado (si te da mucho más, hay fuga entre entrenamiento y validación o estás estandarizando distinto).
- **T decide.** Con T = 10 (1 s) la ventana ve a lo sumo una transición y no sabe cuánto duró la fase: casi no distingue `shuffle_blocks`. Con T = 40 ya cabe un bloque de 3 s completo; con T = 80, cualquier ventana contiene alguna combinación que en `structured` no existe.
- **¿Por qué se queda en 0.76 con T = 40?** El 24 % restante son ventanas de `shuffle_blocks` que *también* aparecen en `structured` (por ejemplo, 4 s seguidos de avance dentro de una fase de 6 s). No son anómalas: el modelo hace bien en no marcarlas.
- **AE frente a PCA.** La red no lineal separa mejor con el mismo L. Con PCA, un latente grande (L = 16) reconstruye también parte de lo anómalo y la separación cae a 0.20: es el problema de "generalizar demasiado" del apunte, visible en el modelo lineal. Con estos datos el AE no lo muestra; probá con L = 32 o 64.

Figuras: `act2_fraccion_T.png` (fracción contra T) y `act2_error_T40.png` (error contra el tiempo, T = 40, en los tres modos; escala simétrica logarítmica).

### Actividad 3 (avanzada): LSTM frente a persistencia y AR (≈ 2 h)

```bash
python3 u6_activity3.py                        # unos 3 min
python3 u6_activity3.py --T 10 70 --seeds 0    # versión rápida
```

Entrena un `LSTMPredictor` (32 unidades) que predice x_t a partir de las T muestras anteriores, con T = 10 y T = 70, y lo compara con persistencia (repetir el último valor) y con un AR(p) lineal multicanal, con p elegido por validación entre 1 y 60 (siempre p ≤ T). Reporta el **skill** = 1 − MSE_modelo / MSE_persistencia.

**Qué deberías ver** (`act3_resumen.csv`, promedio de tres semillas):

| Modelo | T | structured | shuffle_blocks | iid |
|---|---|---|---|---|
| LSTM | 10 | 0.04 | −0.06 | 0.13 |
| LSTM | 70 | 0.82 | −14.1 | −0.19 |
| AR(10) | 10 | 0.04 | −0.02 | 0.03 |
| AR(60) | 70 | 1.00 | −200 | −18.5 |

- **Con T = 10 nadie le gana a la persistencia** en `structured`: para anticipar una transición hay que saber cuánto lleva la fase (hasta 60 muestras), y eso no está en la entrada.
- **Con T = 70 la LSTM aprende el ciclo**: skill 0.82, y en `shuffle_blocks` anticipa transiciones que no ocurren, así que su error se dispara. Ese error es la señal de detección.
- **El AR lineal de orden alto también lo aprende, y mejor.** `structured` sintético es perfectamente periódico y determinístico, y un AR(60) lo predice casi sin error. Esto ilustra lo que el apunte cita de las comparaciones de detectores: un método simple bien elegido puede igualar o superar al aprendido. La ventaja de la LSTM, si existe, aparece con datos reales, donde las duraciones varían y una regla lineal rígida se vuelve frágil (fijate qué tan negativo es el skill del AR(60) fuera de lo nominal: −200 en `shuffle_blocks`, frente a −14 de la LSTM).
- En `iid`, con T = 10, la LSTM tiene skill positivo: aprendió a predecir la media, que en ruido independiente le gana a la persistencia (Unidad 4: el MSE de la persistencia es 2σ² y el de la media σ²).

Figura: `act3_prediccion.png`, 30 s de `structured` con la señal, la persistencia y la predicción de la LSTM para cada T. Mirá dónde se equivoca cada una.

### Actividad 4: el error en el detector y evaluación por eventos (≈ 2 h)

Primero entrená y guardá los modelos que vas a comparar (en total, un par de minutos):

```bash
python3 train_model.py --kind ae   --T 40 --out models/ae_T40
python3 train_model.py --kind ae   --T 40 --counts --out models/ae_T40_nk   # con canal n_k
python3 train_model.py --kind lstm --T 70 --out models/lstm_T70
python3 u6_activity4.py --models models/ae_T40 models/ae_T40_nk models/lstm_T70
```

`u6_activity4.py` hace, para cada configuración (solo los indicadores de la Unidad 5, cada modelo solo, y la Unidad 5 más cada modelo):

1. los indicadores de la Unidad 5 (`H_v`, `Hcond_v`, `H_dt`, `mse_v`, `mse_w`) con el mismo código que los nodos, a 2 Hz y ventanas de 5 s;
2. el error del modelo sobre la grilla, **promediado en la misma ventana causal de 5 s**;
3. calibración robusta (mediana y 1.4826·MAD), z saturado en 10, τ en el cuantil 0.99 del score sobre `test_structured.csv` (nominal y **distinta** de la de entrenamiento), y K = 3 consecutivos;
4. la evaluación por eventos sobre `eventos.csv`.

**Qué deberías ver** (`act4_resumen.csv` y `act4_eventos.csv`; retardo en s, "---" = no detectado):

| Configuración | Segmento shuffle (150 s) | Flood (300 s) | Replay (450 s) | Falsas alarmas |
|---|---|---|---|---|
| Unidad 5 | 7.5 | 1.5 | 1.5 | 5 |
| ae_T40 solo | 2.5 | --- | 1.5 | 0 |
| ae_T40_nk solo | 2.5 | 1.5 | 1.5 | 0 |
| lstm_T70 solo | 2.5 | --- | 1.5 | 0 |
| Unidad 5 + ae_T40 | 2.5 | 1.5 | 1.5 | 4 |
| Unidad 5 + ae_T40_nk | 2.5 | 1.5 | 1.5 | 4 |
| Unidad 5 + lstm_T70 | 2.5 | 1.5 | 1.5 | 0 |

Cómo leerla:

- **El modelo ve la pérdida de estructura antes**: 2.5 s frente a 7.5 s de los indicadores de la Unidad 5.
- **Sin el canal `n_k`, el modelo no ve el flood.** El re-muestreo se queda con el último mensaje de cada celda, y los valores del atacante son los legítimos: la grilla es idéntica a la nominal. Con `n_k` (4 o 5 mensajes por celda en lugar de 1) lo detecta. Los indicadores de la Unidad 5 lo ven por `H_dt`.
- **El replay por reemplazo** se detecta solo por las costuras (al empezar y al terminar), igual que con los indicadores explícitos. Mirá en `act4_score.png` cómo el score vuelve a nominal en el medio del evento.
- **Combinar no es gratis.** El score de la Unidad 5 tiene sus propias falsas alarmas, y sumarle un indicador muy separado no siempre las quita: con `ae_T40` siguen 4. Con 10 minutos de datos, "5 falsas alarmas" es una estimación muy gruesa (unas 30 por hora); para comparar en serio necesitás corridas nominales más largas.
- **El error de la LSTM es "picudo"**: casi cero, con saltos en cada transición nominal. Por eso el script lo promedia en 5 s. Probá `--agg max --agg-sec 0.5` y vas a ver que el z satura en cada transición y τ queda pegado a `z_max`.

Si no conocés los eventos (por ejemplo, en el proyecto integrador), usá `--events ""`: el script lista los intervalos de alarma de cada configuración en `act4_alarmas.csv`, sin evaluación por eventos.

La columna `indicador_principal` de `act4_eventos.csv` dice qué indicador aportó más al score en el primer instante de alarma de cada evento: usala para atribuir cada detección.

---

## 6. El modelo en línea en ROS 2 (opcional)

El nodo `model_error` hace en línea lo mismo que la actividad 4 fuera de línea: arma la grilla con un temporizador, mantiene las últimas T muestras, evalúa el modelo y publica en `/model_metrics` (`std_msgs/Float64MultiArray`):

```
data = [t, n_ventanas, err_medio, err_ultimo]
```

donde `err_medio` es el promedio del error en los últimos 5 s (el indicador) y `err_ultimo` el de la última ventana.

**Instalación** (dentro del contenedor, con los nodos de la Unidad 5 ya instalados):

```bash
cd ~/project/codigos/u6
cp cmdvel_tools_nodes/model_error.py cmdvel_tools_nodes/csv_logger.py cmdvel_tools_nodes/cmdvel_msg.py \
   u6_data.py u6_models.py \
   ~/project/ros2_ws/src/cmdvel_tools/cmdvel_tools/      # csv_logger.py reemplaza al de la Unidad 5
# agregar la línea de setup_entry_points.txt en setup.py
cd ~/project/ros2_ws && colcon build --packages-select cmdvel_tools && source install/setup.bash
```

**Uso**, con el generador y el registrador de la Unidad 5:

```bash
ros2 run cmdvel_tools cmd_vel_generator --ros-args -p use_sim_time:=true -p mode:=structured
ros2 run cmdvel_tools model_error --ros-args -p use_sim_time:=true -p model:=$HOME/project/codigos/u6/models/ae_T40
ros2 topic echo /model_metrics
```

El `csv_logger` de esta unidad registra lo mismo que el de la Unidad 5 y, además, `model_metrics.csv` (`t, n_ventanas, err_medio, err_ultimo`):

```bash
ros2 run cmdvel_tools csv_logger --ros-args -p use_sim_time:=true -p log_dir:=outputs_u6/online
```

**Comprobación:** el `err_medio` en línea tiene que coincidir con el que calcula `u6_activity4.py` sobre el CSV de la misma corrida, salvo diferencias de alrededor del 10 % instante a instante, porque el temporizador del nodo no tiene la misma fase que la grilla fuera de línea. Lo verificamos con los tres modelos: los promedios por evento coinciden al tercer decimal.

> El detector `anomaly_detector` de la Unidad 5 lee solo `/entropy_metrics` y `/cmd_vel_pred_metrics`. Integrar `/model_metrics` al detector en línea es una extensión optativa: agregá una suscripción y el indicador a `INDICATORS`, y calibrá con `calibrate_detector.py` sobre un registro que incluya la nueva columna.

> **La fase del temporizador.** Si el temporizador del nodo cae justo sobre los instantes de llegada de los mensajes, `n_k` alterna entre 0 y 2 aun sin ataque. Con el canal `n_k`, revisá en el CSV que en operación nominal `n_k` sea casi siempre 1; si no, reiniciá el nodo (cambia la fase) o cambiá un poco `rate_hz` del generador.

---

## 7. Usar tus propios registros

Cualquier CSV del logger de la Unidad 1 (columnas `t_sec, v, w`) sirve. Para Gazebo:

1. Registrá al menos 10 min de `structured` para entrenar, 5 min más en otra corrida para calibrar y probar, y las corridas de `shuffle_blocks` e `iid`.
2. Copialos en una carpeta con los mismos nombres de la sección 4 y pasá `--data-dir` a las actividades 1 a 3.
3. Para la actividad 4, usá el atacante de la Unidad 5 (`cmdvel_attacker`) y su `events_file`. Los tiempos de `eventos.txt` tienen que estar **relativos al primer mensaje del CSV**: si el atacante escribe tiempos absolutos (sim time), restale el `t_sec` del primer mensaje.

Lo que vas a notar con datos reales: las transiciones caen una muestra antes o después según la sincronización del reloj, y el modelo ya no llega a error cero en lo nominal. El umbral sube, y la separación entre modos baja. Medilo, no lo supongas.

---

## 8. Referencia: archivos, columnas y parámetros

### Archivo de modelo (`train_model.py`)

`models/<nombre>.pt` contiene los pesos; `models/<nombre>.json`, todo lo demás:

| Campo | Significado |
|---|---|
| `kind`, `T`, `hidden`, `latent` | arquitectura (`ae` o `lstm`) |
| `channels` | `["v", "w"]` o `["v", "w", "n_k"]` |
| `err` | indicador del AE: `mean` (toda la ventana) o `last` (última muestra) |
| `rate_hz` | frecuencia de la grilla |
| `standardizer` | media y desvío por canal, **solo de entrenamiento** |
| `val_error` | mediana, escala robusta y cuantil 0.99 del error en validación |
| `epochs`, `val_loss`, `seed`, `train_csv` | trazabilidad |

### Parámetros principales

| Script | Parámetro | Por defecto | Qué controla |
|---|---|---|---|
| `train_model.py` | `--kind` | `ae` | `ae` o `lstm` |
| | `--T` | 40 | longitud de ventana (muestras de la grilla) |
| | `--latent` | 4 | dimensión del código (solo `ae`) |
| | `--counts` | no | agregar el canal `n_k` |
| | `--err` | `mean` | `last` reacciona antes ante anomalías breves |
| | `--frac-train` | 0.8 | partición por bloques de tiempo |
| `u6_activity2.py` | `--T`, `--seeds`, `--latent-sweep` | 10 40 80 / 0 1 2 / 2 4 16 | barridos |
| `u6_activity3.py` | `--T`, `--ar-orders` | 10 70 / 1…60 | ventanas y órdenes de AR |
| `u6_activity4.py` | `--K`, `--q`, `--z-max` | 3 / 0.99 / 10 | persistencia, cuantil de τ, saturación |
| | `--agg`, `--agg-sec` | `mean` / 5 | agregación del error del modelo |
| | `--tolerance` | 5 | segundos tras un evento en que una alarma todavía cuenta |
| `simulate_cmdvel_csv.py` | `--mode`, `--block-len` | `structured` / 30 | modo del generador |
| | `--segment-mode/-at/-len` | — | segmento anómalo dentro de la corrida |

El entrenamiento usa Adam (tasa 10⁻³, lotes de 64), hasta 200 épocas con parada temprana (paciencia 15) sobre la pérdida de validación, y se queda con los pesos de la mejor época.

---

## 9. Entregables y checklist

Cada grupo entrega el código y los archivos `models/*.json` usados, las tablas y figuras de las actividades 1 a 4 (la 3 es optativa) y un informe de 2 a 3 páginas que responda las preguntas guía del apunte con evidencia propia.

- [ ] Datos re-muestreados en grilla regular, con los huecos excluidos de las ventanas.
- [ ] Estandarización con estadísticas solo de entrenamiento, guardadas con el modelo.
- [ ] Partición por bloques de tiempo y prueba nominal en una corrida distinta.
- [ ] Actividad 1 que muestre el límite del modelo puntual.
- [ ] Barrido de T con al menos tres semillas, y comparación con PCA.
- [ ] Umbral fijado en validación y fracción sobre τ medida en la prueba nominal.
- [ ] Evaluación por eventos con y sin el error del modelo en el score, con y sin `n_k`.
- [ ] Informe con las preguntas guía respondidas.

---

## 10. Resolución de problemas

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `[error] Falta PyTorch` | no está instalado en ese Python | sección 3; verificá con `python3 -c "import torch"` |
| pip descarga varios GB | instalaste la versión con CUDA | desinstalá y usá `--index-url .../whl/cpu` |
| `numpy` 2.x rompe algo de ROS 2 | pip lo actualizó al instalar algo | `pip3 install --ignore-installed "numpy==1.26.4"` |
| `No matching distribution found for torch` | la versión pedida no existe para tu Python (el contenedor Humble usa 3.10) | usá la de la imagen: `pip3 install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu` |
| En la prueba nominal la fracción sobre τ es ≫ 0.01 | fuga (ventanas mezcladas antes de partir) o estandarización distinta | usá `split_blocks` y el mismo `Standardizer` |
| `shuffle_blocks` da error distinto de `structured` en la actividad 1 | estadísticas de estandarización distintas por modo | estandarizá todo con las de entrenamiento |
| `[aviso] ... parece re-muestreado` | le pasaste un CSV de la Unidad 2 | usá el CSV original del logger |
| τ queda en 10.00 en la actividad 4 | el error satura el z en lo nominal (error "picudo") | `--agg mean --agg-sec 5` (por defecto) |
| Ningún evento detectado en datos propios | tiempos de `eventos.txt` absolutos | restales el primer `t_sec` del CSV |
| Resultados distintos en cada corrida | semillas o hilos | `--seeds` fijas; los decimales pueden variar entre máquinas |
| El entrenamiento tarda mucho | CPU lenta | `--seeds 0`, menos valores de T, o `torch.set_num_threads` |

---

## 11. Referencia rápida

```bash
python3 -c "import torch; print(torch.__version__)"     # ya viene en la imagen
python3 generate_datasets.py
python3 u6_activity1.py
python3 u6_activity2.py
python3 u6_activity3.py                    # optativa
python3 train_model.py --kind ae   --T 40 --out models/ae_T40
python3 train_model.py --kind ae   --T 40 --counts --out models/ae_T40_nk
python3 train_model.py --kind lstm --T 70 --out models/lstm_T70
python3 u6_activity4.py --models models/ae_T40 models/ae_T40_nk models/lstm_T70
```

**Lección clave.** Un modelo aprendido no ve más que lo que le mostrás: el autoencoder puntual mira la distribución, la ventana ve el orden solo hasta la escala que cubre, y el ritmo de los mensajes, que el re-muestreo descarta, vuelve solo si lo agregás como canal. Antes de ajustar la red, elegí la representación y medí contra una línea base simple.
