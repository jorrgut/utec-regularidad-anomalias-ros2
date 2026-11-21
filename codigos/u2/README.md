# Unidad 2: Análisis temporal, ventanas deslizantes y re-muestreo

**Guía práctica de laboratorio**
Optativa *Análisis de Regularidad y Detección de Anomalías en Sistemas Robóticos*, Ingeniería en Mecatrónica (UTEC)

Esta guía acompaña al apunte de la Unidad 2. Parte del CSV que registraste en la Unidad 1 y te lleva hasta las dos entregas: el análisis por ventanas (Actividad 1) y el re-muestreo con comparación de métodos (Actividad 2). La teoría está en el apunte: por qué el jitter no alarga las ventanas, a dónde va el jitter al re-muestrear y por qué el ZOH es causal. Acá está el **cómo**.

---

## Índice

1. [Qué vas a hacer](#1-qué-vas-a-hacer)
2. [Requisitos](#2-requisitos)
3. [Instalación de los scripts](#3-instalación-de-los-scripts)
4. [Los scripts de la unidad](#4-los-scripts-de-la-unidad)
5. [Flujo completo](#5-flujo-completo)
6. [Cómo leer los resultados](#6-cómo-leer-los-resultados)
7. [Usar `u2_tools` en tus propios scripts](#7-usar-u2_tools-en-tus-propios-scripts)
8. [Actividad 1: ventanas sobre los datos originales](#8-actividad-1-ventanas-sobre-los-datos-originales)
9. [Actividad 2: re-muestreo y comparación de métodos](#9-actividad-2-re-muestreo-y-comparación-de-métodos)
10. [Practicar con datos sintéticos](#10-practicar-con-datos-sintéticos)
11. [Entregables y checklist](#11-entregables-y-checklist)
12. [Resolución de problemas](#12-resolución-de-problemas)
13. [Referencia rápida](#13-referencia-rápida)

---

## 1. Qué vas a hacer

```
 data/cmdvel_u1.csv ──┬──► window_features.py ──► features por ventana (datos originales)
   (de la Unidad 1)   │      (samples / time)       ¿cuánto dura cada ventana? ¿cuántas muestras tiene?
                      │
                      └──► resample_cmdvel_20hz.py ──► data/cmdvel_20hz.csv ──► window_features.py
                             (linear / previous)       grilla uniforme,            features comparables
                                                       huecos en NaN
```

Al final vas a poder responder, con tus propios datos:

- ¿Las ventanas por muestras duran todas lo mismo? Si no, ¿qué evento lo explica: jitter, pérdidas o pausas?
- ¿Qué cambia, y qué no, cuando re-muestreás a 20 Hz?
- ¿Qué método de interpolación conviene para un comando de velocidad, y por qué?

**Tiempo estimado de laboratorio:** unas 6 h.

---

## 2. Requisitos

- El entorno de la Unidad 1: contenedor Docker con Python, numpy 1.26, pandas y matplotlib. **No hace falta ROS** para esta unidad, ni tampoco `scipy`.
- El archivo `data/cmdvel_u1.csv` que registraste en la Unidad 1. Si no lo tenés o salió mal, ver la [sección 10](#10-practicar-con-datos-sintéticos).

> Como no usa ROS, esta unidad también se puede trabajar fuera del contenedor, en cualquier computadora con Python 3:
> `pip install "numpy<2" pandas matplotlib`

---

## 3. Instalación de los scripts

No hay que instalar ni copiar nada: los scripts están en `~/project/codigos/u2`. Se ejecutan **desde esa carpeta**, porque importan `u2_tools.py`, que está junto a ellos. Traé a esta carpeta el registro de la Unidad 1:

```bash
cd ~/project/codigos/u2
mkdir -p data
cp ../u1/data/cmdvel_u1.csv data/
python3 window_features.py --help        # verificación
python3 resample_cmdvel_20hz.py --help
```

```
codigos/u2/
├── u2_tools.py                  ← funciones del apunte (las importan los otros scripts)
├── resample_cmdvel_20hz.py
├── window_features.py
├── simulate_cmdvel_csv.py       ← el mismo de la Unidad 1
├── data/
│   └── cmdvel_u1.csv            ← tu registro de la Unidad 1 (copiado)
└── outputs_u2/                  ← se crea solo
```

---

## 4. Los scripts de la unidad

| Archivo | Qué hace |
|---|---|
| `u2_tools.py` | Las funciones del apunte, sección "Implementación práctica en Python": `prepare_series`, `clip_physical`, `hampel_filter`, `resample_to_grid`, `sliding_windows`, `time_windows`, `window_stats`, `verify_resampling` |
| `resample_cmdvel_20hz.py` | Re-muestrea `v` y `w` a una grilla uniforme, marca los huecos y verifica el resultado |
| `window_features.py` | Calcula descriptores por ventana (por muestras o por tiempo) y resume cómo varían duración y cantidad de muestras |
| `simulate_cmdvel_csv.py` | Genera CSV sintéticos con pérdidas, pausas y retrasos conocidos |

### `resample_cmdvel_20hz.py`

| Parámetro | Por defecto | Significado |
|---|---|---|
| `--csv` | (obligatorio) | CSV de entrada con `t_sec, v, w` |
| `--out` | `data/cmdvel_20hz.csv` | CSV de salida |
| `--freq` | 20 | Frecuencia de la grilla (Hz) |
| `--method` | `linear` | `linear`, `nearest` o `previous` (ZOH) |
| `--max-gap` | 3 × mediana(Δt) | Hueco máximo tolerable (s); los puntos dentro de huecos mayores quedan vacíos |
| `--no-clip` | (apagado) | No descartar valores fuera de los límites físicos del burger (\|v\| ≤ 0.22 m/s, \|ω\| ≤ 2.84 rad/s) |
| `--outdir` | `outputs_u2` | Carpeta para figuras y reporte |
| `--plot-from` / `--plot-len` | primer cambio de `v` / 15 | Tramo de la señal que se grafica (s desde el inicio) |

**Salidas:**

| Archivo | Contenido |
|---|---|
| `data/cmdvel_20hz.csv` | Columnas `t_sec, v, w, dt, gap`. En los huecos, `v` y `w` quedan vacíos y `gap` vale 1. |
| `outputs_u2/dt_before_after.png` | Histograma de Δt original y de la grilla, en la misma escala |
| `outputs_u2/signal_compare_<método>.png` | Señal original (círculos) y re-muestreada (línea) en un tramo, con los huecos marcados |
| `outputs_u2/resample_report_<método>.txt` | Muestras descartadas, timestamps repetidos, huecos y resultado de la verificación |

### `window_features.py`

| Parámetro | Modo | Significado |
|---|---|---|
| `--csv` | ambos | CSV con `t_sec` y las señales (original o re-muestreado) |
| `--mode` | ambos | `samples` (por cantidad de muestras) o `time` (por duración) |
| `--n` / `--stride` | `samples` | Tamaño de ventana y paso, en muestras (paso por defecto: n/2) |
| `--tw` / `--step` | `time` | Ancho de ventana y paso, en segundos (paso por defecto: tw/2) |
| `--signals` | ambos | Columnas a analizar (por defecto `v w`) |
| `--out` | ambos | CSV de salida |
| `--plot` | ambos | Guarda una figura con el mismo nombre que el CSV, en `.png` |

**Columnas del CSV de salida:** `i, t_start, t_end, T_span, n_samples, n_nan` y, para cada señal, `<señal>_mean, _std, _median, _mad, _min, _max, _range, _power`.

- `T_span`: duración entre la primera y la última muestra de la ventana, (n−1)·Δ en la grilla.
- `n_nan`: cantidad de valores vacíos (huecos) dentro de la ventana. Si es mayor que 0, la ventana atraviesa un hueco.

---

## 5. Flujo completo

```bash
cd ~/project/codigos/u2

# Paso 1: ventanas por muestras sobre los datos originales (≈ 2 s a 10 Hz)
python3 window_features.py --csv data/cmdvel_u1.csv --mode samples \
    --n 20 --stride 10 --out data/features_samples.csv --plot

# Paso 2: ventanas por tiempo sobre los datos originales
python3 window_features.py --csv data/cmdvel_u1.csv --mode time \
    --tw 2.0 --step 1.0 --out data/features_time.csv --plot

# Paso 3: re-muestrear a 20 Hz con los dos métodos
python3 resample_cmdvel_20hz.py --csv data/cmdvel_u1.csv \
    --method linear --out data/cmdvel_20hz.csv
python3 resample_cmdvel_20hz.py --csv data/cmdvel_u1.csv \
    --method previous --out data/cmdvel_20hz_zoh.csv

# Paso 4: ventanas sobre la señal re-muestreada (≈ 2 s a 20 Hz)
python3 window_features.py --csv data/cmdvel_20hz.csv --mode samples \
    --n 40 --stride 20 --out data/features_samples_20hz.csv --plot
python3 window_features.py --csv data/cmdvel_20hz.csv --mode time \
    --tw 2.0 --step 1.0 --out data/features_time_20hz.csv
```

> Cada script imprime un resumen en la terminal. **Leelo:** ahí está la mitad de lo que tenés que interpretar.

---

## 6. Cómo leer los resultados

### Resumen de `window_features.py` en modo `samples`

```
T_span: mediana 1.9010 s | min 1.8410 | max 3.9060 | CV 8.769%
Ventanas con T_span > 1.05 × mediana: 56  (posibles pérdidas o pausas)
  -> CV > 5%: re-muestreá o usá ventanas por tiempo.
```

| Lo que ves | Qué significa |
|---|---|
| `T_span` casi igual en todas las ventanas (CV < 1 %) | Muestreo regular. **El jitter solo nunca da un CV alto**: no se acumula (apunte, sección "¿Qué hace variar la duración de una ventana?"). |
| Ventanas marcadas con `T_span` ≈ mediana + 0.1 s | Ventanas que contienen **un mensaje perdido**: cada pérdida suma un período |
| Una o pocas ventanas con `T_span` muy grande | Una **pausa** |
| CV alto con muchas ventanas marcadas | Pérdidas frecuentes o un publicador irregular, como la teleoperación |

### Resumen en modo `time`

```
Muestras por ventana: mediana 20 | min 1 | max 21 | CV 7.99%
Ventanas con < 80% de la mediana de muestras: 3  (posibles pérdidas o pausas)
```

- Con jitter solamente, el número de muestras varía en ±1 alrededor de la mediana: es normal.
- Ventanas con muchas menos muestras indican una pausa o varias pérdidas seguidas.

### Reporte de `resample_cmdvel_20hz.py`

```
Puntos en huecos (NaN): 42 (0.70%) en 1 hueco(s)
Verificación:
  v: OK
  w: OK
```

- **Puntos en huecos:** cada hueco que la auditoría de la Unidad 1 mostró como un Δt grande tiene que aparecer acá. Si en la Unidad 1 tenías una pausa y acá dice 0 huecos, revisá `--max-gap`.
- **Verificación:** si algo dice `FALLÓ`, leé el mensaje. Ver [Resolución de problemas](#12-resolución-de-problemas).

### Las figuras

| Figura | Qué mirar |
|---|---|
| `dt_before_after.png` | A la izquierda, la distribución real de Δt; a la derecha, una sola barra en 0.05 s. **Esa barra no es una mejora**: la grilla es así por construcción. |
| `signal_compare_linear.png` | En las transiciones (por ejemplo, de 0.20 a 0), la línea pasa por valores intermedios que la señal original nunca tomó |
| `signal_compare_previous.png` | Con ZOH, la línea siempre vale lo mismo que el último círculo: no aparecen valores nuevos |
| `features_*.png` | Arriba, la duración o la cantidad de muestras de cada ventana; abajo, media y desviación estándar de `v`. Buscá si las ventanas marcadas coinciden con cambios en los estadísticos. |

> En una transición, ver un punto de la grilla con un valor intermedio entre 0.20 y 0 no es un error: es exactamente la conversión de jitter en error de amplitud que describe el apunte (e ≈ −ẋ·δ).

---

## 7. Usar `u2_tools` en tus propios scripts

Las Actividades 1 y 2 piden **tus** scripts (`u2_activity1.py`, `u2_activity2.py`). Podés escribirlos desde cero o importar las funciones del apunte:

```python
import numpy as np
import pandas as pd
from u2_tools import resample_to_grid, sliding_windows, window_stats, verify_resampling

df = pd.read_csv('data/cmdvel_u1.csv')
t, v = df['t_sec'].to_numpy(), df['v'].to_numpy()

# Re-muestrear con dos métodos
tg, v_lin = resample_to_grid(t, v, target_freq=20.0, method='linear')
_,  v_zoh = resample_to_grid(t, v, target_freq=20.0, method='previous')
verify_resampling(t, v, tg, v_lin)

# ¿Dónde difieren? Solo en las transiciones
diff = np.abs(v_lin - v_zoh)
print('puntos con diferencia > 1e-9:', np.sum(diff > 1e-9))

# Ventanas de 100 puntos (5 s) con paso 50
W, starts = sliding_windows(v_lin, 100, 50)
st = window_stats(W)
print('ventanas:', len(W), '| con huecos:', np.sum(st['n_nan'] > 0))
print('T_span:', tg[starts[0] + 99] - tg[starts[0]], 's')   # 4.95 s
```

> Si usás estas funciones, **decilo en el informe**. Lo que se evalúa es que entiendas y justifiques lo que hacen, no que las reescribas.

---

## 8. Actividad 1: ventanas sobre los datos originales

**Pasos** (detalle en el apunte):

1. Cargar `data/cmdvel_u1.csv` y repetir el análisis de Δt de la Unidad 1. Anotá dónde hay pérdidas y pausas.
2. Ventanas por muestras con n = 50, 100 y 200 y paso n/2:
   ```bash
   for n in 50 100 200; do
     python3 window_features.py --csv data/cmdvel_u1.csv --mode samples \
         --n $n --stride $((n/2)) --out outputs_u2a1/features_n$n.csv --plot
   done
   ```
3. Para cada n: anotá el CV de `T_span` y qué ventanas quedaron marcadas.
4. Relacioná esas ventanas con los eventos del paso 1.
5. Ventanas por tiempo de 2 s con paso 1 s; graficá el número de muestras por ventana.

**Preguntas guía para el informe:**

- ¿El CV de `T_span` baja al aumentar n? ¿Por qué? Pista: la ecuación de duración de ventana del apunte.
- ¿Qué ventanas quedaron marcadas y qué evento contienen?
- ¿En qué casos preferirías ventanas por tiempo?

---

## 9. Actividad 2: re-muestreo y comparación de métodos

**Pasos:**

1. Re-muestrear con `linear` y con `previous` (paso 3 del [flujo](#5-flujo-completo)).
2. Comparar los dos reportes: ¿los mismos huecos?, ¿verificación OK en ambos?
3. Graficar el mismo tramo con ambos métodos. Elegí un tramo con transiciones:
   ```bash
   python3 resample_cmdvel_20hz.py --csv data/cmdvel_u1.csv --method linear \
       --out data/cmdvel_20hz.csv --plot-from 5 --plot-len 15
   python3 resample_cmdvel_20hz.py --csv data/cmdvel_u1.csv --method previous \
       --out data/cmdvel_20hz_zoh.csv --plot-from 5 --plot-len 15
   ```
4. Cuantificar la diferencia entre métodos **por separado** en transiciones y en tramos constantes (con tu script; ver el ejemplo de la [sección 7](#7-usar-u2_tools-en-tus-propios-scripts)).
5. Ventanas de n = 100 y paso 50 sobre la grilla. Verificá que `T_span` = 4.95 s en todas.
6. Contar cuántas ventanas tienen `n_nan > 0`.

**Preguntas guía para el informe:**

- Que todos los Δt valgan 0.05 s, ¿significa que se eliminó el jitter? ¿A dónde fue?
- ¿Dónde se concentran las diferencias entre `linear` y `previous`? ¿Por qué ahí?
- ¿Por qué `T_span` da 4.95 s y no 5.0 s?
- ¿Qué método usarías para las unidades siguientes? Justificalo.

---

## 10. Practicar con datos sintéticos

Si tu CSV de la Unidad 1 no tiene eventos interesantes, o si querés comprobar que interpretás bien las figuras, generá uno con eventos **conocidos**:

```bash
# 1 % de pérdidas, pausa de 2 s en t = 150 s, 5 retrasos aislados, reloj cuantizado a 1 ms
python3 simulate_cmdvel_csv.py --out data/sint_eventos.csv \
    --drop-prob 0.01 --pause-at 150 --pause-len 2.0 --n-delays 5 --quantum 0.001

python3 window_features.py --csv data/sint_eventos.csv --mode samples \
    --n 20 --stride 10 --out data/sint_features.csv --plot
python3 resample_cmdvel_20hz.py --csv data/sint_eventos.csv --method previous \
    --out data/sint_20hz_zoh.csv --outdir outputs_sint --plot-from 145 --plot-len 12
```

**Lo que deberías encontrar:**

- En `sint_features.png`, un pico grande de `T_span` cerca de t = 150 s (la pausa) y muchas ventanas marcadas con `T_span` ≈ 2.0 s (las pérdidas).
- En `signal_compare_previous.png`, el hueco de 150 a 152 s marcado en rojo.
- Con `--drop-prob 0` y sin pausa, el CV de `T_span` tiene que quedar muy por debajo del 1 %, **aunque el jitter siga ahí**.

---

## 11. Entregables y checklist

### Actividad 1

- Script `u2_activity1.py`
- Carpeta `outputs_u2a1/` con figuras
- Informe técnico (10–15 líneas)

### Actividad 2

- Script `u2_activity2.py`
- `cmdvel_u1_resampled_20hz.csv`, con el método que elijas como principal
- Carpeta `outputs_u2a2/` con figuras
- Informe técnico (12–18 líneas)

### Archivos del flujo de la [sección 5](#5-flujo-completo)

- `data/features_samples.csv` y `data/features_time.csv`
- `data/cmdvel_20hz.csv` y `data/cmdvel_20hz_zoh.csv`
- `outputs_u2/dt_before_after.png`
- (Recomendado) `data/features_samples_20hz.csv` y `data/features_time_20hz.csv`

### ✅ Checklist antes de entregar

- [ ] Identifiqué las pérdidas y pausas de mis datos, y las relacioné con las ventanas marcadas
- [ ] Reporté el CV de `T_span` para cada n y lo expliqué **sin** atribuirlo al jitter si no corresponde
- [ ] Re-muestreé con al menos dos métodos y con control de huecos
- [ ] No afirmé que el re-muestreo "elimina" o "reduce" el jitter
- [ ] Comparé los métodos en transiciones y en tramos constantes por separado
- [ ] Justifiqué la elección de método y de tamaño de ventana con **mis** datos
- [ ] Las figuras tienen ejes con unidades, títulos y leyendas
- [ ] Mis scripts corren de principio a fin con un solo comando

### Qué se evalúa

| Criterio | Insuficiente | Aceptable | Destacado |
|---|---|---|---|
| Comprensión conceptual | Confunde ventanas por muestras y por tiempo, o afirma que el re-muestreo elimina el jitter | Aplica ventanas y re-muestreo correctamente pero sin explicar sus límites | Explica qué información no gana (o inventa) el re-muestreo, a dónde va el jitter, y por qué importa |
| Justificación técnica | Elige tamaño de ventana, frecuencia o método sin argumento | Justifica con una regla general (n ≥ 30, factor 2×) | Justifica con evidencia de sus propios datos (CV de `T_span`, diferencias entre métodos en transiciones) |
| Análisis crítico | No compara antes/después, o reporta números sin interpretarlos | Compara estadísticos antes/después y entre métodos | Compara, interpreta las diferencias y discute los trade-offs de cada elección |

---

## 12. Resolución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'u2_tools'` | `u2_tools.py` no está en la carpeta desde donde ejecutás | `cd ~/project/codigos/u2` y verificá que `u2_tools.py` esté ahí |
| `[error] faltan columnas en el CSV` | El CSV no es el del logger de la Unidad 1 | Revisá el encabezado con `head -2 data/cmdvel_u1.csv`: tiene que tener `t_sec,v,w` |
| `Hay timestamps que retroceden` | Se reinició Gazebo durante el registro de la Unidad 1 | Volvé a registrar, o recortá el CSV antes del reinicio |
| El reporte dice `timestamps repetidos resueltos: N` | Normal con tiempo simulado (mismo paso de simulación) | Nada que hacer: se queda con el último mensaje de cada instante |
| `0 hueco(s)` aunque en la Unidad 1 viste una pausa | `--max-gap` demasiado grande | Usá el valor por defecto o uno menor; comparalo con la mediana de Δt |
| Demasiados huecos (muchos pequeños) | `--max-gap` demasiado chico, o pérdidas muy frecuentes | Probá `--max-gap 0.35`; si persiste, el problema son los datos |
| Verificación `FALLÓ -> La media cambió X%` | Muchos puntos en huecos, o un método que distorsiona una señal con pocas muestras | Revisá los huecos; compará con `--method previous` |
| Verificación `FALLÓ -> Mínimo/Máximo fuera de rango` | No debería pasar con los tres métodos | Revisá si modificaste `u2_tools.py` |
| `descartadas por límites físicos: N` con N grande | El generador publicó velocidades mayores al máximo del burger | Revisá `--v-fwd`/`--w-turn` en la Unidad 1; o usá `--no-clip` si es intencional |
| `la serie tiene X muestras, menos que n` | Ventana más larga que el registro | Usá un n menor o registrá más tiempo |
| Las figuras no se generan dentro del contenedor | No debería pasar: los scripts no abren ventanas | Verificá permisos de escritura en `outputs_u2/` |

---

## 13. Referencia rápida

```bash
# Re-muestrear
python3 resample_cmdvel_20hz.py --csv data/cmdvel_u1.csv --method linear   --out data/cmdvel_20hz.csv
python3 resample_cmdvel_20hz.py --csv data/cmdvel_u1.csv --method previous --out data/cmdvel_20hz_zoh.csv

# Ventanas por muestras / por tiempo
python3 window_features.py --csv <csv> --mode samples --n 40 --stride 20 --out <salida.csv> --plot
python3 window_features.py --csv <csv> --mode time --tw 2.0 --step 1.0 --out <salida.csv> --plot

# Datos sintéticos
python3 simulate_cmdvel_csv.py --out data/sint.csv --drop-prob 0.01 --pause-at 150 --pause-len 2

# Ayuda de cualquier script
python3 <script>.py --help
```

**Equivalencias útiles a 20 Hz:** n = 20 → 1 s · n = 40 → 2 s · n = 100 → 5 s · n = 200 → 10 s (`T_span` = (n−1) × 0.05 s).

---

### Lecturas para profundizar

- VanderPlas (2018), *Understanding the Lomb–Scargle periodogram*: una introducción clara al análisis de series irregulares sin re-muestrear.
- McKinney (2022), *Python for Data Analysis*, capítulo sobre series temporales en pandas.
- Keogh y Lin (2005), *Clustering of time-series subsequences is meaningless*: por qué las ventanas solapadas requieren cuidado.
- La bibliografía completa, organizada por tema, está al final del apunte de la unidad.
