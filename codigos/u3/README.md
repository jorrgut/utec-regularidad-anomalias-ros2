# Unidad 3: Entropía de Shannon como medida de regularidad

**Guía práctica de laboratorio**
Optativa *Análisis de Regularidad y Detección de Anomalías en Sistemas Robóticos*, Ingeniería en Mecatrónica (UTEC)

Esta guía acompaña al apunte de la Unidad 3. Parte de los CSV re-muestreados de la Unidad 2 y te lleva hasta las dos entregas: la entropía por ventana (Actividad 1) y la demostración de que la entropía no ve el orden temporal (Actividad 2). La teoría está en el apunte: por qué la grilla de bins tiene que ser fija, por qué el estimador está sesgado y qué mide la entropía condicional. Acá está el **cómo**.

---

## Índice

1. [Qué vas a hacer](#1-qué-vas-a-hacer)
2. [Requisitos e instalación](#2-requisitos-e-instalación)
3. [Los scripts de la unidad](#3-los-scripts-de-la-unidad)
4. [Flujo completo](#4-flujo-completo)
5. [Cómo leer los resultados](#5-cómo-leer-los-resultados)
6. [Usar `u3_tools` en tus propios scripts](#6-usar-u3_tools-en-tus-propios-scripts)
7. [Actividad 1: entropía en ventanas](#7-actividad-1-entropía-en-ventanas)
8. [Actividad 2: distribución vs. orden temporal](#8-actividad-2-distribución-vs-orden-temporal)
9. [Practicar con datos sintéticos (incluye un ataque)](#9-practicar-con-datos-sintéticos-incluye-un-ataque)
10. [Entregables y checklist](#10-entregables-y-checklist)
11. [Resolución de problemas](#11-resolución-de-problemas)
12. [Referencia rápida](#12-referencia-rápida)

---

## 1. Qué vas a hacer

```
 data/cmdvel_20hz.csv ─────┬──► entropy_windows.py ──► H(v), H(ω) por ventana, comparación con σ
 data/cmdvel_20hz_zoh.csv ─┘     (--compare)           y entre re-muestreo lineal y ZOH
 (de la Unidad 2)
                           └──► bins_sensitivity.py ─► ¿cómo cambia H con el número de bins?

 data/cmdvel_u1.csv ─────────► entropy_dt.py ────────► entropía de Δt: regularidad del muestreo
 (ORIGINAL, de la Unidad 1)

 tu script u3_activity2.py ──► señal periódica vs. desordenada: H, H(X_t+1|X_t), autocorrelación
```

Al final vas a poder responder, con tus propios datos:

- ¿Cuándo es alta la entropía de la velocidad, y qué está haciendo el robot en esos momentos?
- ¿Qué aporta la entropía que no aporte la desviación estándar?
- ¿Qué tipo de anomalía ve la entropía de los valores, cuál ve la entropía de Δt, y cuál no ve ninguna de las dos?

**Tiempo estimado de laboratorio:** unas 5 h.

---

## 2. Requisitos e instalación

- El mismo entorno de la Unidad 2: Python con `numpy<2`, `pandas` y `matplotlib`. **No hace falta ROS ni `scipy`.**
- De la Unidad 1: `data/cmdvel_u1.csv`.
- De la Unidad 2: `data/cmdvel_20hz.csv` (lineal) y `data/cmdvel_20hz_zoh.csv` (ZOH). Si no los tenés, generalos:
  ```bash
  python3 ../u2/resample_cmdvel_20hz.py --csv data/cmdvel_u1.csv --method linear   --out data/cmdvel_20hz.csv
  python3 ../u2/resample_cmdvel_20hz.py --csv data/cmdvel_u1.csv --method previous --out data/cmdvel_20hz_zoh.csv
  ```

Los scripts están en `~/project/codigos/u3` y se ejecutan **desde esa carpeta** (importan `u3_tools.py`). Traé los datos de las unidades anteriores:

```bash
cd ~/project/codigos/u3
mkdir -p data
cp ../u2/data/cmdvel_20hz.csv ../u2/data/cmdvel_20hz_zoh.csv data/
cp ../u1/data/cmdvel_u1.csv data/
python3 entropy_windows.py --help        # verificación
```

---

## 3. Los scripts de la unidad

| Archivo | Qué hace | Entrada |
|---|---|---|
| `u3_tools.py` | Funciones del apunte: `make_edges`, `entropy`, `entropy_sliding_windows`, `conditional_entropy`, `autocorr`, más `binary_entropy` y `spearman` | — |
| `entropy_windows.py` | Entropía, entropía condicional, media, σ y MAD por ventana; figuras y resumen | CSV **re-muestreado** |
| `bins_sensitivity.py` | Repite el cálculo con varios números de bins y compara valor absoluto y orden relativo | CSV **re-muestreado** |
| `entropy_dt.py` | Entropía de los intervalos Δt por ventanas de tiempo | CSV **original** de la Unidad 1 |
| `simulate_cmdvel_csv.py` | CSV sintéticos con pérdidas, pausas, retrasos e **inyección de mensajes** (`--flood-*`) | — |

### `entropy_windows.py`

| Parámetro | Por defecto | Significado |
|---|---|---|
| `--csv` | (obligatorio) | CSV re-muestreado |
| `--signals` | `v w` | Señales a analizar |
| `--n` / `--stride` | 100 / n/2 | Tamaño de ventana y paso, en muestras (100 muestras = 5 s a 20 Hz) |
| `--nbins` | 15 | Número de bins de la grilla **fija** |
| `--range` | límites físicos del burger | Rango de la grilla, por ejemplo `--range v=-0.22,0.22 w=-2.84,2.84` |
| `--max-nan` | 0 | Fracción máxima de huecos (NaN) tolerada por ventana; las demás se descartan |
| `--miller-madow` | (apagado) | Corrección de sesgo del estimador |
| `--compare` / `--labels` | — | Segundo CSV (por ejemplo, ZOH) y nombres para la leyenda |
| `--out` | `outputs_u3a1/entropy_windows.csv` | Tabla de salida; las figuras van a la misma carpeta |

**Salidas:** `entropy_windows.csv` (columnas `t_start, t_center, nan_frac` y, por señal, `_H, _Hnorm, _Hcond, _mean, _std, _mad`), `entropy_series.png`, `entropy_vs_std.png`, `entropy_summary.txt` y, con `--compare`, `entropy_compare.png` y `entropy_windows_compare.csv`.

### `bins_sensitivity.py`

| Parámetro | Por defecto | Significado |
|---|---|---|
| `--signal` | `v` | Una señal por vez |
| `--nbins` | `5 10 15 20 25` | Números de bins a probar (siempre sobre el mismo rango) |
| `--ref` | el del medio | Configuración de referencia para la correlación de Spearman |
| `--outdir` | `outputs_u3a1` | Carpeta de salida |

**Salidas:** `bins_sensitivity.csv`, `bins_spearman.csv` (matriz) y `bins_sensitivity.png`.

### `entropy_dt.py`

| Parámetro | Por defecto | Significado |
|---|---|---|
| `--csv` | (obligatorio) | CSV **original** de la Unidad 1 |
| `--tw` / `--step` | 10 / tw/2 | Ventanas por **tiempo**, en segundos |
| `--nbins` | 20 | Bins de la grilla sobre Δt |
| `--dt-max` | 3 × mediana(Δt) | Borde superior de la grilla; los Δt mayores (pausas) caen en el último bin |
| `--miller-madow` | (apagado) | Recomendado aquí: el número de mensajes por ventana varía |

**Salidas:** `entropy_dt.csv` (columnas `t_start, n_msgs, H_dt, median_dt, n_long`) y `entropy_dt.png`.

---

## 4. Flujo completo

```bash
cd ~/project/codigos/u3

# 1. Entropía por ventana, comparando re-muestreo lineal y ZOH
python3 entropy_windows.py --csv data/cmdvel_20hz.csv \
    --compare data/cmdvel_20hz_zoh.csv --labels linear ZOH \
    --out outputs_u3a1/entropy_windows.csv

# 2. Sensibilidad al número de bins (una señal por vez)
python3 bins_sensitivity.py --csv data/cmdvel_20hz.csv --signal v --outdir outputs_u3a1
python3 bins_sensitivity.py --csv data/cmdvel_20hz.csv --signal w --outdir outputs_u3a1/w

# 3. Entropía de Δt sobre los datos ORIGINALES
python3 entropy_dt.py --csv data/cmdvel_u1.csv --miller-madow --out outputs_u3a1/entropy_dt.csv
```

> Cada script imprime un resumen en la terminal y lo guarda en un `.txt` o `.csv`. **Leelo:** ahí está buena parte de lo que tenés que interpretar.

---

## 5. Cómo leer los resultados

### Resumen de `entropy_windows.py`

```
[v] grilla: 15 bins en [-0.22, 0.22] m/s  (máximo posible: 3.91 bits)
  H: media 0.597 | std 0.420 | min 0.000 | max 1.117 bits
  Ventanas con H = 0 (un solo bin ocupado): 27
  Correlación H vs. std: Pearson 0.995 | Spearman 0.988
  Nota: la señal toma solo 2 valores distintos. ...
```

| Lo que ves | Qué significa |
|---|---|
| Ventanas con H = 0 | El robot estuvo toda la ventana en un mismo estado (siempre detenido o siempre avanzando) |
| Picos de H ≈ 1 bit | Ventanas que contienen una transición: la mitad del tiempo en un estado y la mitad en otro, h₂(0.5) = 1 bit |
| H algo **mayor** que 1 bit con la señal lineal | La interpolación lineal creó valores intermedios en las transiciones (Unidad 2), que ocupan bins extra |
| Correlación H vs. σ cercana a 1 | Esperable con el generador: con dos niveles, H y σ dependen de la misma fracción de tiempo (apunte, "Una consecuencia de que la señal tenga dos niveles"). **No es un error.** |
| Correlación baja, o puntos lejos de la curva | La señal tiene más de dos niveles (teleoperación, valores intermedios): ahí la entropía sí aporta información distinta de σ |
| H máxima muy por debajo de log₂(N_bins) | Normal: la señal usa pocos de los bins disponibles |

### Las figuras

| Figura | Qué mirar |
|---|---|
| `entropy_series.png` | ¿Los picos de H coinciden con los cambios de fase de v(t)? Los huecos (ventanas descartadas) aparecen como cortes en la curva. |
| `entropy_vs_std.png` | Los puntos y la curva discontinua de dos niveles. Los que quedan por encima tienen valores intermedios en la ventana. |
| `entropy_compare.png` | Dónde difieren lineal y ZOH: casi solo en las ventanas con transiciones. ZOH no inventa valores, así que su H nunca supera la de dos niveles. |
| `bins_sensitivity.png` | Izquierda: H media según el número de bins, y el máximo posible. Derecha: correlación de rangos con la configuración de referencia. |
| `entropy_dt.png` | Arriba Δt, al medio H(Δt), abajo los mensajes por ventana. Los puntos rojos son ventanas con H(Δt) atípicamente alta. |

### Sensibilidad a los bins: dos resultados posibles

- **Con el generador, H casi no cambia con el número de bins.** La señal toma pocos valores exactos y, una vez que cada uno cae en su propio bin, refinar la grilla no cambia nada. Es lo que predice la ecuación de la entropía diferencial del apunte para señales "discretas".
- **Con teleoperación, o con la señal lineal en ventanas con muchas transiciones, H crece con el número de bins**, aproximadamente 1 bit cada vez que se duplica. Ahí el valor absoluto depende de la grilla, y solo tiene sentido comparar ventanas con la misma grilla.

En ambos casos, mirá el panel derecho: si la correlación de Spearman se mantiene alta (> 0.9), el **orden relativo** de las ventanas no depende de la grilla, y tus conclusiones comparativas son robustas.

### Entropía de Δt

| Lo que ves | Qué significa |
|---|---|
| H(Δt) baja y estable | Muestreo regular |
| H(Δt) alta, con **más** mensajes por ventana | Inyección de mensajes (*flooding*): aparecen intervalos muy cortos y variables |
| H(Δt) algo alta, con **menos** mensajes | Pérdidas o una pausa |
| Un pico aislado de Δt arriba, sin cambio grande en H | Una pausa larga cae en un solo bin (el último): suma poca entropía. Se ve mejor en el número de mensajes. |

> Como las ventanas por tiempo tienen distinto número de mensajes, el sesgo del estimador cambia de una ventana a otra. Por eso conviene `--miller-madow`, y por eso hay que mirar H(Δt) **junto con** `n_msgs`.

---

## 6. Usar `u3_tools` en tus propios scripts

Las Actividades 1 y 2 piden **tus** scripts. Podés importar las funciones del apunte:

```python
import numpy as np
import pandas as pd
from u3_tools import make_edges, entropy, entropy_sliding_windows, conditional_entropy, autocorr

df = pd.read_csv('data/cmdvel_20hz.csv')
v = df['v'].to_numpy()

edges_v = make_edges(-0.22, 0.22, 15)                     # grilla FIJA, una sola vez
H_v, starts = entropy_sliding_windows(v, 100, 50, edges_v)
t_center = (starts + 50) * 0.05                            # centro de cada ventana (s)

print(f'ventanas: {len(H_v)} | descartadas por huecos: {np.isnan(H_v).sum()}')
print(f'H media: {np.nanmean(H_v):.3f} bits')
```

> Si usás estas funciones, **decilo en el informe**. Lo que se evalúa es que entiendas y justifiques lo que hacen.

**Nunca uses `np.histogram(x, bins=15)` sin bordes fijos** para calcular entropías por ventana: cada ventana tendría su propia grilla y los valores no serían comparables. En el apunte está la demostración: un comando constante con un poco de ruido da 3.8 bits en lugar de 0.

---

## 7. Actividad 1: entropía en ventanas

1. Correr el [flujo completo](#4-flujo-completo).
2. En `entropy_series.png`, relacionar los picos de H(v) y H(ω) con las fases del generador (avanzar, detenerse, girar).
3. Identificar las ventanas de mayor entropía (están en el resumen) y mirar qué pasa en la señal en esos tiempos.
4. Analizar `bins_sensitivity.png` para v y para ω.
5. Analizar `entropy_vs_std.png`: ¿los puntos siguen la curva de dos niveles? ¿Cuáles se apartan y por qué?
6. Comparar lineal contra ZOH en `entropy_compare.png` y en el resumen.
7. (Recomendado) Analizar `entropy_dt.png`.

**Preguntas guía para el informe:**

- ¿Cuándo es alta la entropía en tus datos? Relacionalo con lo que hace el robot.
- ¿El número de bins cambia el valor absoluto? ¿Y el orden relativo de las ventanas? Explicá por qué, con la ecuación de la entropía diferencial.
- ¿La entropía aporta información distinta de σ **en tus datos**? Si la correlación es ≈ 1, explicá por qué y en qué situación dejaría de serlo.
- ¿Cuánto cambia la entropía entre lineal y ZOH? ¿Qué parte de la entropía refleja al robot y qué parte al método de re-muestreo?

---

## 8. Actividad 2: distribución vs. orden temporal

Esta actividad es **tu** script (`u3_activity2.py`). El apunte tiene todo el código paso a paso, y con `u3_tools` es corto:

```python
import numpy as np
from u3_tools import make_edges, entropy, conditional_entropy, autocorr

pattern = ([0.2]*20 + [0.0]*20 + [0.1]*20 + [0.0]*20) * 3
signal_A = np.array(pattern[:200])
signal_B = signal_A.copy()
np.random.default_rng(42).shuffle(signal_B)

edges = make_edges(0.0, 0.2, 10)
for name, s in (('A', signal_A), ('B', signal_B)):
    print(name, f'H = {entropy(s, edges):.3f}', f'Hcond = {conditional_entropy(s, edges):.3f}')
rA, rB = autocorr(signal_A, 80), autocorr(signal_B, 80)
```

**Valores esperados** (para que verifiques tu implementación):

| | H (bits) | H(X_t+1 \| X_t) (bits) | Autocorrelación en el retardo 1 |
|---|---|---|---|
| Señal A (periódica) | 1.485 | ≈ 0.29 | ≈ 0.91 |
| Señal B (desordenada) | 1.485 | ≈ 1.48 | ≈ 0 |

Lo que falta es lo importante: las figuras (histogramas, series, autocorrelaciones) y la **interpretación**.

**Preguntas guía para el informe:**

- ¿Por qué las entropías de Shannon **tienen que** ser iguales?
- ¿Qué capturan la entropía condicional y la autocorrelación que la entropía no captura?
- ¿Qué ataque o falla pasaría inadvertido si solo se usara la entropía de los valores? ¿Qué indicador lo detectaría? Pista: la [sección 9](#9-practicar-con-datos-sintéticos-incluye-un-ataque).

---

## 9. Practicar con datos sintéticos (incluye un ataque)

El simulador ahora puede **inyectar mensajes**: durante un intervalo agrega mensajes extra con valores legítimos, como haría un atacante que quiere pasar desapercibido.

```bash
# Pérdidas, una pausa en t = 150 s y una inyección de 40 msg/s entre t = 220 y 230 s
python3 simulate_cmdvel_csv.py --out data/sint_ataque.csv \
    --drop-prob 0.01 --pause-at 150 --pause-len 2.0 --quantum 0.001 \
    --flood-at 220 --flood-len 10 --flood-rate 40

python3 resample_cmdvel_20hz.py --csv data/sint_ataque.csv --out data/sint_ataque_20hz.csv \
    --outdir outputs_sint
python3 entropy_windows.py --csv data/sint_ataque_20hz.csv --out outputs_sint/entropy_windows.csv
python3 entropy_dt.py --csv data/sint_ataque.csv --miller-madow --out outputs_sint/entropy_dt.csv
```

**Lo que deberías encontrar:**

- En `outputs_sint/entropy_series.png`, **nada especial entre 220 y 230 s**: los valores inyectados son legítimos, así que la distribución de cada ventana no cambia.
- En `outputs_sint/entropy_dt.png`, un **salto claro de H(Δt)** y del número de mensajes en esas mismas ventanas.
- La pausa de 150 s aparece como ventanas descartadas en `entropy_windows` y como menos mensajes en `entropy_dt`.

**Conclusión para el informe:** la entropía de los valores y la entropía de Δt miran cosas distintas. Un detector que use solo una de las dos tiene puntos ciegos.

---

## 10. Entregables y checklist

### Actividad 1

- Script `u3_activity1.py` (puede llamar a los scripts de esta carpeta o importar `u3_tools`)
- Carpeta `outputs_u3a1/` con:
  - series temporales de H(v) y H(ω) junto con la señal;
  - gráfico de sensibilidad al número de bins;
  - gráfico de dispersión de entropía contra desviación estándar;
  - comparación entre lineal y ZOH.
- Informe técnico (10–15 líneas)

### Actividad 2

- Script `u3_activity2.py`
- Carpeta `outputs_u3a2/` con histogramas, series, autocorrelaciones y la tabla de H y H(X_t+1 | X_t)
- Informe técnico (8–12 líneas)

### ✅ Checklist antes de entregar

- [ ] Usé una **grilla de bins fija** (mismos bordes) en todas las ventanas que comparo
- [ ] Todas las ventanas que comparo tienen el **mismo número de muestras**; las que tienen huecos están descartadas o señaladas
- [ ] Verifiqué mi función de entropía con los ejemplos a mano del apunte
- [ ] Hice el experimento de sensibilidad y comenté **valor absoluto** y **orden relativo** por separado
- [ ] Interpreté la relación entre H y σ en mis datos, sin asumir que "deberían" ser distintas
- [ ] Comparé lineal contra ZOH
- [ ] En la Actividad 2 fijé la semilla y reporté H, la entropía condicional y la autocorrelación
- [ ] No escribí "alta entropía = aleatorio"
- [ ] Las figuras tienen ejes con unidades, títulos y leyendas

### Qué se evalúa

| Criterio | Insuficiente | Aceptable | Destacado |
|---|---|---|---|
| Comprensión conceptual | Trata la entropía como sinónimo de aleatoriedad, o no distingue distribución de orden temporal | Calcula e interpreta la entropía correctamente, sin explicar su limitación central | Explica con sus datos por qué histogramas idénticos pueden esconder comportamientos distintos, y qué medida lo revela |
| Justificación técnica | Usa rangos de bins distintos por ventana, o compara entropías con distintas grillas | Usa una grilla fija y justifica N_bins con una regla general | Justifica la grilla con su experimento de sensibilidad: qué cambia (valor absoluto) y qué no (orden relativo) |
| Análisis crítico | No conecta los resultados con implicaciones prácticas | Reconoce que la entropía no distingue las señales A y B | Explica qué ataque o falla pasaría inadvertido y qué indicador complementario (condicional, de Δt, autocorrelación) lo detectaría |

---

## 11. Resolución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'u3_tools'` | `u3_tools.py` no está en la carpeta desde donde ejecutás | `cd ~/project/codigos/u3` y verificá que el archivo esté ahí |
| `[aviso] ... el paso temporal no es constante` | Le pasaste a `entropy_windows.py` el CSV original | Usá el re-muestreado de la Unidad 2 (`cmdvel_20hz.csv`) |
| `entropy_dt.py`: `todos los Δt son iguales` | Le pasaste el CSV re-muestreado | Usá el **original** (`cmdvel_u1.csv`) |
| `[aviso] ... valores fuera de [...]` | La señal supera los límites físicos del burger | Revisá los datos; si es intencional, ampliá con `--range` |
| Muchas ventanas con H = `NaN` | Ventanas que tocan huecos (pausas) | Es lo esperado. Si son demasiadas, revisá el `--max-gap` de la Unidad 2 o usá `--max-nan 0.05` |
| H = 0 en casi todas las ventanas | Ventanas más cortas que las fases del generador, o grilla con muy pocos bins | Probá `--n 200`; verificá que 0 y 0.2 caigan en bins distintos |
| H con valores cercanos a log₂(N_bins) en ventanas casi constantes | Calculaste con `np.histogram(x, bins=N)` sin bordes fijos | Usá `make_edges` y `entropy` de `u3_tools` |
| Correlación H vs. σ ≈ 1 | **No es un error**: señal de dos niveles | Explicalo en el informe (apunte, "Una consecuencia de que la señal tenga dos niveles") |
| La entropía condicional da negativa | Error de implementación (se restó al revés) | Usá `conditional_entropy` de `u3_tools` como referencia: H(X,Y) − H(X) ≥ 0 |
| Resultados distintos cada vez en la Actividad 2 | No fijaste la semilla del mezclado | `np.random.default_rng(42)` |

---

## 12. Referencia rápida

```bash
# Entropía por ventana (y comparación lineal / ZOH)
python3 entropy_windows.py --csv data/cmdvel_20hz.csv --compare data/cmdvel_20hz_zoh.csv \
    --labels linear ZOH --out outputs_u3a1/entropy_windows.csv

# Sensibilidad al número de bins
python3 bins_sensitivity.py --csv data/cmdvel_20hz.csv --signal v --nbins 5 10 15 20 25

# Entropía de Δt (sobre el CSV ORIGINAL)
python3 entropy_dt.py --csv data/cmdvel_u1.csv --tw 10 --step 5 --miller-madow

# Datos sintéticos con un ataque de inyección
python3 simulate_cmdvel_csv.py --out data/sint.csv --flood-at 220 --flood-len 10 --flood-rate 40
```

**Valores de referencia:** h₂(0.5) = 1 bit · h₂(0.1) = 0.469 bits · log₂(15) = 3.91 bits · sesgo del estimador con n = 100 y 15 bins ocupados ≈ −0.10 bits.

---

### Lecturas para profundizar

- Cover y Thomas (2006), *Elements of Information Theory*, capítulo 2: la referencia estándar.
- MacKay (2003), *Information Theory, Inference, and Learning Algorithms*: más intuitivo, disponible gratis en la web del autor.
- Bandt y Pompe (2002), *Permutation entropy*: cuatro páginas sobre cómo incorporar el orden temporal.
- La bibliografía completa, organizada por tema, está al final del apunte de la unidad.
