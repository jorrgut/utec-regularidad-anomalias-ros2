# Unidad 4: Predictibilidad y error de predicción

**Guía práctica de laboratorio**
Optativa *Análisis de Regularidad y Detección de Anomalías en Sistemas Robóticos*, Ingeniería en Mecatrónica (UTEC)

Esta guía acompaña al apunte de la Unidad 4. En esta unidad la cátedra entrega los scripts de las actividades: tu trabajo es **ejecutarlos, entender cada salida y escribir el análisis**. La teoría está en el apunte: por qué la predictibilidad depende del predictor, por qué σ se estima en un período de referencia y qué significa la alineación causal. Acá está el **cómo**.

---

## Índice

1. [Qué vas a hacer](#1-qué-vas-a-hacer)
2. [Requisitos e instalación](#2-requisitos-e-instalación)
3. [Los archivos de la unidad](#3-los-archivos-de-la-unidad)
4. [Actividad 1: predicción a un paso](#4-actividad-1-predicción-a-un-paso)
5. [Actividad 2: Rolling NLL y detección](#5-actividad-2-rolling-nll-y-detección)
6. [Actividad 3: entropía y predictibilidad](#6-actividad-3-entropía-y-predictibilidad)
7. [Actividad 4: tus datos de /cmd_vel y un ataque de replay](#7-actividad-4-tus-datos-de-cmd_vel-y-un-ataque-de-replay)
8. [Usar `u4_tools` en tus propios análisis](#8-usar-u4_tools-en-tus-propios-análisis)
9. [Entregables y checklist](#9-entregables-y-checklist)
10. [Resolución de problemas](#10-resolución-de-problemas)
11. [Referencia rápida](#11-referencia-rápida)

---

## 1. Qué vas a hacer

```
 Actividades 1-3: señales sintéticas con propiedades conocidas
   u4_activity1.py ──► ¿qué tan bien predicen la persistencia y un AR(p)?
   u4_activity2.py ──► detectar un tramo anómalo con Rolling NLL, contra la verdad conocida
   u4_activity3.py ──► entropía vs. error de predicción: los cuatro cuadrantes

 Actividad 4: datos del robot (de las Unidades 1 y 2)
   data/cmdvel_20hz_zoh.csv ──► predict_cmdvel.py ──► error, NLL conjunto (v + ω), alarmas
   simulate_cmdvel_csv.py --replay-* ──► el mismo análisis ante un ataque de replay
```

Al final vas a poder responder, con resultados propios:

- ¿Una señal es "predecible", o es predecible **para un modelo dado**?
- ¿Cuándo el NLL aporta algo distinto del MSE?
- ¿Qué ataques ve el error de predicción, cuáles ve la entropía y cuáles no ve ninguno?

**Tiempo estimado de laboratorio:** unas 5 h, sumando las cuatro actividades y los informes.

---

## 2. Requisitos e instalación

- El entorno de las unidades anteriores: Python con numpy 1.26, pandas y matplotlib. **No hace falta ROS ni `scipy`.**
- Para la Actividad 4: `data/cmdvel_20hz_zoh.csv` y `data/cmdvel_20hz.csv` (Unidad 2), y los scripts de re-muestreo (Unidad 2) y de entropía (Unidad 3), que se llaman desde sus carpetas (`../u2/`, `../u3/`).

Los scripts están en `~/project/codigos/u4` y se ejecutan **desde esa carpeta** (todos los `.py` de la unidad se importan entre sí). Para la Actividad 4, traé los CSV re-muestreados de la Unidad 2:

```bash
cd ~/project/codigos/u4
python3 u4_activity1.py --help           # verificación
mkdir -p data
cp ../u2/data/cmdvel_20hz.csv ../u2/data/cmdvel_20hz_zoh.csv data/
```

Los scripts de otras unidades se llaman con su ruta (por ejemplo `python3 ../u2/resample_cmdvel_20hz.py`): Python busca sus módulos (`u2_tools.py`) en la carpeta del script, así que funcionan sin copiar nada.

> Los scripts **no abren ventanas**: guardan las figuras en archivos `.png` dentro de `outputs_u4aN/`. Abrilas con el visor de imágenes de tu computadora; la carpeta `codigos/u4/` se ve también desde fuera del contenedor, dentro de `utec_optativa_ros2`.

> Esta versión **reemplaza** a los scripts anteriores de la unidad (`u4_utils.py`, `run_all_activities.py`). Si los tenés, borralos para no mezclar versiones.

---

## 3. Los archivos de la unidad

| Archivo | Qué hace |
|---|---|
| `u4_tools.py` | Funciones del apunte: persistencia, `fit_ar`/`predict_ar`, métricas, `estimate_sigma`, `nll_gauss`, ventanas causales, umbrales, `evaluate_detection`. Además, utilidades de entropía de la Unidad 3. |
| `generate_signals.py` | Señales sintéticas: Structured, Shuffled, IID, Periodic, Slow y la señal con anomalía de la Actividad 2 |
| `u4_activity1.py` | Actividad 1 → `outputs_u4a1/` |
| `u4_activity2.py` | Actividad 2 → `outputs_u4a2/` |
| `u4_activity3.py` | Actividad 3 → `outputs_u4a3/` |
| `predict_cmdvel.py` | Actividad 4: error de predicción y NLL conjunto sobre un CSV re-muestreado |
| `simulate_cmdvel_csv.py` | El simulador de las unidades anteriores, ahora con **replay** (`--replay-*`) |
| `u4_plotstyle.py` | Estilo común de las figuras |

### Las señales sintéticas

Todas tienen semilla fija: **cada vez que corras los scripts vas a obtener exactamente los mismos números.**

| Señal | Cómo es | Para qué sirve |
|---|---|---|
| Structured | Bloques: 20 muestras en 0.2 m/s, 20 en 0 (ciclo de 40) | Como el generador de la Unidad 1 |
| Shuffled | Los mismos valores de Structured, en orden aleatorio | Misma distribución, sin orden |
| IID | Valores independientes, uniformes en [0, 0.2] | Sin ninguna estructura temporal |
| Periodic | 0.1 + 0.1·sin(2πt/8): cambia en todos los pasos | Entropía alta y patrón aprendible |
| Slow | Senoide lenta, período de 100 muestras | Entropía alta y predecible incluso con persistencia |

---

## 4. Actividad 1: predicción a un paso

```bash
python3 u4_activity1.py
```

Para cada señal compara **persistencia** con **AR(p)**, con p = 1, 2, 4 y 8. El AR se **entrena en la primera mitad** y se **evalúa en la segunda**; todas las métricas son de la mitad de prueba.

### Lo que vas a ver en la terminal

```
  Structured  persistencia RMSE 0.0439 (NRMSE 0.44) | mejor AR: AR(8)  RMSE 0.0423 (NRMSE 0.42) | skill +0.073
  Shuffled    persistencia RMSE 0.1440 (NRMSE 1.44) | mejor AR: AR(1)  RMSE 0.1000 (NRMSE 1.00) | skill +0.518
  IID         persistencia RMSE 0.0847 (NRMSE 1.41) | mejor AR: AR(1)  RMSE 0.0600 (NRMSE 1.00) | skill +0.498
  Periodic    persistencia RMSE 0.0540 (NRMSE 0.76) | mejor AR: AR(2)  RMSE 0.0000 (NRMSE 0.00) | skill +1.000

Verificación de MSE_persistencia = V^2 * r (señales de dos niveles):
  Structured  tasa de cambios r = 0.0482 | V^2 r = 0.001928 | MSE medido = 0.001928
```

### Cómo leerlo

| Lo que ves | Qué significa |
|---|---|
| **NRMSE** = RMSE / desvío de la señal | ≈ 1: el predictor no supera a "predecir siempre la media". ≈ 0: predice casi perfecto. |
| **skill** = 1 − MSE_AR / MSE_persistencia | Cuánto mejora el AR a la persistencia: 0 nada, 1 perfecto |
| Skill ≈ 0.5 en IID y Shuffled | **No significa que sean predecibles.** La persistencia es *peor* que la media en señales sin estructura (MSE = 2σ²), y el AR aprende a predecir la media (MSE = σ²). Mirá el NRMSE: queda en 1. |
| Periodic: AR(2) con error 0 | Toda senoide muestreada cumple una recurrencia lineal de orden 2: dos valores alcanzan |
| Structured: skill ≈ 0.07 | Con 8 valores pasados no se puede anticipar una transición que llega cada 20 muestras |
| V²·r = MSE medido | La ecuación del apunte: con dos niveles, la persistencia mide la **tasa de cambios** |

### Archivos generados en `outputs_u4a1/`

| Archivo | Qué mirar |
|---|---|
| `metrics_comparison.csv` | Todas las métricas, por señal y predictor |
| `signals_comparison.png` | Las cuatro señales (primeras 160 muestras) |
| `error_series.png` | Error de persistencia (color) y del mejor AR (gris discontinuo) en la mitad de prueba. En Structured: picos solo en las transiciones. |
| `rmse_by_order.png` | RMSE de prueba según el orden; la línea punteada de cada color es la persistencia. Buscá el **menor orden** a partir del cual el error deja de bajar. |
| `error_autocorr.png` | ¿Los errores de persistencia tienen estructura? En Structured, picos de signo alternado cada 20 muestras (las transiciones); en IID y Shuffled, una correlación de −0.5 en el retardo 1. |

---

## 5. Actividad 2: Rolling NLL y detección

```bash
python3 u4_activity2.py
```

Señal de bloques (ciclo de 40 muestras) con un tramo anómalo **conocido** entre las muestras 500 y 700. Predictor: persistencia.

1. Estima σ en el **período de referencia** (muestras 0–300) y la deja **fija**.
2. Calcula Rolling MSE y Rolling NLL **causales** (cada ventana se asigna a su instante final) con N = 25, 40, 50 y 100.
3. Calibra el umbral en la referencia: μ + 3σ y, para comparar, el robusto (mediana + 3·1.4826·MAD).
4. Exige 3 ventanas seguidas sobre el umbral para alarmar.
5. Evalúa contra la verdad conocida.

### Lo que vas a ver en la terminal

```
     N umbral           retardo  muestras en alarma antes  vuelve a normal
    25 mu+k*sigma            15                         0              +15
    25 robusto (MAD)          2                        69             +285
       (MAD de referencia = 0: el indicador tiene muchos empates)
    40 mu+k*sigma             7                         0              +34
    ...
Correlación Rolling MSE vs. Rolling NLL (σ fija, N = 50): 1.0000
```

### Cómo leerlo

| Columna o línea | Qué significa |
|---|---|
| `retardo` | Muestras entre el inicio real (500) y la primera alarma. Con alineación causal es **siempre ≥ 0**. |
| `muestras en alarma antes` | Falsas alarmas antes de la anomalía. Con μ + 3σ debería dar 0. |
| `vuelve a normal` | Cuántas muestras después de la 700 sigue la alarma: la ventana tarda en "olvidar" los errores anómalos. Crece con N. |
| Umbral robusto con cientos de falsas alarmas | En la referencia, cada ventana tiene un número entero de transiciones: el indicador toma 2 o 3 valores, la **MAD da 0** y el umbral queda pegado a la mediana. La robustez de la mediana protege contra valores atípicos, no contra distribuciones discretas. |
| N = 40 y N = 100 se comportan "bien" con ambos umbrales | En esas ventanas el número de transiciones es siempre el mismo (hay una cada 20 muestras), así que el indicador de referencia es **constante**. |
| Correlación MSE–NLL = 1.0000 | Con σ fija, el NLL es una transformación lineal del MSE: **detectan exactamente lo mismo**. |

### Archivos generados en `outputs_u4a2/`

| Archivo | Qué mirar |
|---|---|
| `rolling_nll_analysis.png` | Cuatro paneles (señal, error, Rolling MSE, Rolling NLL) con la anomalía sombreada, el umbral y las alarmas en rojo. La línea punteada vertical marca el fin de la referencia. |
| `window_size_comparison.png` | Rolling NLL con cada N: ventanas cortas reaccionan rápido pero oscilan; ventanas largas son suaves pero tardan en subir y en bajar |
| `sigma_comparison.png` | Arriba, σ fija (correcto); abajo, σ estimada en cada ventana (el error común): el indicador pasa a ser ≈ ½ + ½·ln(2πσ²_ventana) y pierde la comparación con lo normal |
| `detection_stats.csv` | La tabla de la terminal, con más columnas |

**Experimentá:** `python3 u4_activity2.py --k 2 --consec 1` (más sensible: ¿aparecen falsas alarmas?) o `--ref-end 100` (referencia más corta: ¿cambia el umbral?).

---

## 6. Actividad 3: entropía y predictibilidad

```bash
python3 u4_activity3.py
```

Para las cinco señales calcula la entropía de Shannon (10 bins fijos en [0, 0.2] m/s), la entropía condicional y el NRMSE de la persistencia y del mejor AR.

### Lo que vas a ver en la terminal

```
  (H alta si >= 1.66 bits; error alto si NRMSE >= 0.5)
  Structured   0.999  0.278  ...  baja H / bajo error -> baja H / bajo error
  Shuffled     0.999  0.998  ...  baja H / alto error -> baja H / alto error
  IID          3.317  3.177  ...  alta H / alto error -> alta H / alto error
  Periodic     2.500  0.886  ...  alta H / alto error -> alta H / bajo error
  Slow         3.136  0.762  ...  alta H / bajo error -> alta H / bajo error
```

### Cómo leerlo

- **Los cuatro cuadrantes están cubiertos**, y la entropía sola no permite adivinar el error: Structured y Shuffled tienen la misma entropía (≈ 1 bit) y errores opuestos.
- **Periodic cambia de cuadrante según el predictor.** Con persistencia parece impredecible; con AR(2), perfectamente predecible. Es la demostración de que la predictibilidad depende del modelo.
- **Slow no cambia de cuadrante:** es tan suave que hasta la persistencia la predice bien.
- **La entropía condicional** (columna `Hcond`) se parece más al error del mejor AR que la entropía de Shannon: es la medida de la Unidad 3 que "ve" el orden.

### Archivos generados en `outputs_u4a3/`

| Archivo | Qué mirar |
|---|---|
| `entropy_vs_prediction.png` | Un punto lleno (persistencia) y uno hueco (mejor AR) por señal, unidos por una flecha. Las líneas punteadas son los cortes de los cuadrantes. |
| `signal_examples.png` | Las cinco señales, con sus métricas en el título |
| `correlation_matrix.png` | Correlaciones entre métricas. **Son solo 5 señales**: es orientativa, no una conclusión estadística. |
| `comparison_table.csv` | Todos los valores |

---

## 7. Actividad 4: tus datos de /cmd_vel y un ataque de replay

### 7.1 Tus datos

```bash
python3 predict_cmdvel.py --csv data/cmdvel_20hz_zoh.csv --out outputs_u4a4/zoh
python3 predict_cmdvel.py --csv data/cmdvel_20hz.csv     --out outputs_u4a4/linear
```

El script usa el primer tercio del registro como **referencia**: ajusta el AR, estima σ de v y de ω, y calibra el umbral. Calcula el **NLL conjunto** (NLL de v + NLL de ω, cada uno con su σ) y alarma con μ + 3σ y persistencia de 3 ventanas de 5 s.

| Parámetro | Por defecto | Significado |
|---|---|---|
| `--train-frac` | 1/3 | Fracción inicial usada como referencia |
| `--predictor` | `persistencia` | Predictor para el NLL y la detección (`persistencia` o `ar`) |
| `--ar-order` | 4 | Orden del AR (se reporta siempre, aunque no se use para detectar) |
| `--window` / `--k` / `--consec` | 100 / 3 / 3 | Ventana (muestras), umbral μ + kσ, ventanas seguidas para alarmar |
| `--truth INICIO FIN` | — | Intervalo anómalo conocido (s), para evaluar la detección |

**Lo que deberías encontrar con un registro normal del generador:**

| | ZOH | Lineal |
|---|---|---|
| Error de persistencia | Exactamente 0 salvo en las transiciones (unas 4 por ciclo de 13 s) | Distinto de 0 en 2 o 3 muestras por transición (las rampas que crea la interpolación) |
| Skill del AR | ≈ 0: no hay nada que aprender entre transiciones | ≈ 0.4: el AR aprende a continuar las rampas inventadas por la interpolación |
| Alarmas | Ninguna | Ninguna |

> El skill de 0.4 con re-muestreo lineal **no** es información sobre el robot: es el AR aprendiendo una propiedad del método de re-muestreo (Unidad 2). Con ZOH, la señal es fiel a lo que el robot recibió.

### 7.2 Un ataque de replay

El simulador ahora puede **reinyectar comandos grabados**: durante un intervalo reenvía lo que el generador publicó `--replay-shift` segundos antes.

```bash
# Replay "add": los comandos grabados se SUMAN a los legítimos (intercalados)
python3 simulate_cmdvel_csv.py --out data/replay.csv --drop-prob 0.01 --quantum 0.001 \
    --replay-at 200 --replay-len 10 --replay-shift 6.5 --replay-mode add
python3 ../u2/resample_cmdvel_20hz.py --csv data/replay.csv --method previous \
    --out data/replay_20hz_zoh.csv --outdir outputs_u4a4
python3 predict_cmdvel.py --csv data/replay_20hz_zoh.csv --out outputs_u4a4/replay --truth 200 210

# Mismo registro, indicadores de la Unidad 3
python3 ../u3/entropy_windows.py --csv data/replay_20hz_zoh.csv --out outputs_u4a4/replay_entropy.csv
python3 ../u3/entropy_dt.py --csv data/replay.csv --miller-madow --out outputs_u4a4/replay_dt.csv
```

Repetí todo con `--replay-mode replace` (los comandos grabados **reemplazan** a los legítimos; la tasa de mensajes no cambia) y nombres de archivo distintos.

**Lo que deberías encontrar** (con estos parámetros):

| Indicador | Replay `add` | Replay `replace` |
|---|---|---|
| Error de predicción / Rolling NLL | **Detecta**, con ≈ 0.3 s de retardo y sin falsas alarmas: la señal "salta" entre dos secuencias | Solo una alarma en la "costura" final (≈ 210 s): durante el ataque, la secuencia recibida es coherente |
| Entropía de v (Shannon) | No lo distingue: los valores son legítimos y la entropía queda dentro de su rango normal | No lo distingue |
| Entropía condicional de v | **Sube** (de ≈ 0.05 a 0.7–0.9 bits): ve el desorden en el orden | No lo distingue |
| Entropía de Δt | **Sube**: llegan ≈ 50 % más mensajes | No lo distingue: la tasa es la normal |

**Conclusión para el informe:** cada indicador mira algo distinto, y hay ataques, como el replay por reemplazo de una secuencia coherente, que ninguno de estos indicadores ve. Detectarlos exige **otra fuente de información**, por ejemplo comparar los comandos con la odometría. Es un buen punto para discutir los límites de lo que medimos.

---

## 8. Usar `u4_tools` en tus propios análisis

Si querés ir más allá de los scripts (y es bienvenido en el informe):

```python
import numpy as np
from u4_tools import (predict_persistence, fit_ar, predict_ar, estimate_sigma,
                      nll_gauss, rolling_mean_causal, threshold, persistent_alarm,
                      evaluate_detection)
from generate_signals import make_anomaly_signal

x, truth = make_anomaly_signal()
_, e = predict_persistence(x)
e = np.r_[0.0, e]                                # alinear: e[t] es el error al predecir x[t]

sigma = estimate_sigma(e[1:300])                 # σ FIJA, del período de referencia
nll = rolling_mean_causal(nll_gauss(e, sigma), 50)
thr = threshold(nll[49:300], k=3)                # umbral μ + 3σ en la referencia
alarm = persistent_alarm(np.nan_to_num(nll, nan=-np.inf) > thr, 3)
print(evaluate_detection(alarm, 500, 700))       # retardo 19, sin falsas alarmas
```

> Si usás estas funciones, **decilo en el informe**. Lo que se evalúa es que entiendas y justifiques lo que hacen.

---

## 9. Entregables y checklist

### Por actividad

| Actividad | Entregar | Informe |
|---|---|---|
| 1 | `outputs_u4a1/` | 10–15 líneas |
| 2 | `outputs_u4a2/` | 12–18 líneas |
| 3 | `outputs_u4a3/` | 15–20 líneas |
| 4 | `outputs_u4a4/` con los análisis de ZOH, lineal, replay `add` y replay `replace` | 10–15 líneas |

Las preguntas de cada informe están en el apunte.

### ✅ Checklist antes de entregar

- [ ] Reporto el error **de prueba**, no el de entrenamiento
- [ ] Interpreto el skill **junto con** el NRMSE (skill 0.5 en IID no es "predecible")
- [ ] En la Actividad 2, σ es la de la referencia, el retardo es ≥ 0 y comento las falsas alarmas
- [ ] Explico por qué el umbral robusto falla con esta señal
- [ ] Explico por qué Rolling MSE y Rolling NLL detectan lo mismo con σ fija, y cuándo no
- [ ] Muestro que Periodic cambia de cuadrante según el predictor
- [ ] Comparo ZOH contra lineal en mis datos y explico de dónde sale el skill del AR
- [ ] Comparo los indicadores ante el replay `add` y el `replace`
- [ ] Discuto causas no maliciosas antes de hablar de "ataque"
- [ ] Las figuras tienen ejes, unidades y leyendas

### Qué se evalúa

| Criterio | Insuficiente | Aceptable | Destacado |
|---|---|---|---|
| Comprensión conceptual | Asume que alta entropía implica baja predictibilidad, trata la predictibilidad como propiedad de la señal sola, o confunde error instantáneo con anomalía | Calcula e interpreta MSE y NLL correctamente, sin explicar cuándo entropía y predictibilidad divergen | Explica, con Periodic y Slow, por qué ambas métricas son necesarias y por qué el resultado depende del predictor |
| Justificación técnica | Elige umbral, ventana u orden del AR sin argumento, o estima σ en la ventana evaluada | Justifica con reglas generales (μ + 3σ, persistencia de 3 ventanas) | Justifica con evidencia propia: error de prueba según el orden, retardo y falsas alarmas según la ventana, estabilidad del umbral |
| Análisis crítico | Reporta detecciones sin compararlas con la verdad conocida, o acepta retardos negativos | Compara con la verdad conocida y reporta retardo y falsas alarmas | Discute qué causas no maliciosas producirían el mismo patrón, y qué ataques no ve el error de predicción |

---

## 10. Resolución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'u4_tools'` (o `generate_signals`, `u4_plotstyle`) | Los `.py` no están todos en la misma carpeta | `cd ~/project/codigos/u4` y verificá con `ls *.py` |
| No se abre ninguna ventana con gráficos | Es lo esperado: los scripts guardan archivos | Abrí los `.png` de `outputs_u4aN/` |
| `ImportError` al importar `u4_utils` | Tenés scripts de la versión anterior | Borrá `u4_utils.py` y `run_all_activities.py`; usá los de esta carpeta |
| `predict_cmdvel.py`: `[aviso] el paso temporal no es constante` | Le pasaste el CSV original | Usá el re-muestreado de la Unidad 2 |
| `predict_cmdvel.py`: `muy pocas filas completas para ajustar AR` | Registro muy corto o con muchos huecos en el primer tercio | Registrá más tiempo, o subí `--train-frac` |
| Alarmas en tu registro normal | Cambió el comportamiento después de la referencia, o la referencia no contiene ciclos completos | Mirá la figura: ¿la alarma coincide con algo que pasó? Probá `--train-frac 0.5` |
| Ninguna alarma en el replay | Usaste `--replay-mode replace`, o re-muestreaste con un `--max-gap` que tapó el ataque | Con `replace` es lo esperado (sección 7.2); revisá el modo |
| NLL con valores negativos | Es normal: el NLL de una densidad puede ser negativo (σ chica) | Lo que importa es la **comparación** con el umbral, no el signo |
| Números distintos a los de esta guía en las Actividades 1–3 | Modificaste `generate_signals.py` o cambiaste parámetros | Con los valores por defecto, los resultados son exactamente reproducibles |

---

## 11. Referencia rápida

```bash
# Actividades con señales sintéticas
python3 u4_activity1.py                      # --orders 1 2 4 8 16  --n 1000
python3 u4_activity2.py                      # --windows 25 40 50 100  --k 3  --consec 3  --ref-end 300
python3 u4_activity3.py                      # --nbins 10

# Actividad 4
python3 predict_cmdvel.py --csv data/cmdvel_20hz_zoh.csv --out outputs_u4a4/zoh
python3 predict_cmdvel.py --csv data/cmdvel_20hz_zoh.csv --predictor ar --out outputs_u4a4/zoh_ar

# Replay (add o replace) y re-muestreo
python3 simulate_cmdvel_csv.py --out data/replay.csv --replay-at 200 --replay-len 10 \
    --replay-shift 6.5 --replay-mode add
python3 ../u2/resample_cmdvel_20hz.py --csv data/replay.csv --method previous --out data/replay_20hz_zoh.csv

# Todas las actividades sintéticas de una vez
for a in 1 2 3; do python3 u4_activity$a.py; done
```

**Valores de referencia:** persistencia en IID → NRMSE ≈ √2 ≈ 1.41 · predecir la media → NRMSE = 1 · NLL con σ fija: un error de 2σ suma 2 nats y uno de 3σ suma 4.5 nats sobre el mínimo.

---

### Lecturas para profundizar

- Hyndman y Athanasopoulos (2021), *Forecasting: Principles and Practice*, capítulos sobre métodos simples y modelos ARIMA. Libro gratuito en línea.
- Malhotra et al. (2015), *Long short term memory networks for anomaly detection in time series*: la idea de esta unidad con un predictor más potente, como anticipo de la Unidad 6.
- La bibliografía completa, organizada por tema, está al final del apunte de la unidad.
