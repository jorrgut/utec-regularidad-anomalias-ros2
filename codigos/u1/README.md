# Unidad 1: Sistemas robóticos como generadores de datos temporales

**Guía práctica de laboratorio**
Optativa *Análisis de Regularidad y Detección de Anomalías en Sistemas Robóticos*, Ingeniería en Mecatrónica (UTEC)

Esta guía acompaña al apunte de la Unidad 1 y te lleva desde una computadora vacía hasta las dos entregas: los datos registrados (A1) y su análisis temporal (A2). La teoría (por qué la mediana, qué mide Δt, qué es el jitter) está en el apunte. Acá está el **cómo**.

---

## Índice

1. [Qué vas a hacer](#1-qué-vas-a-hacer)
2. [Requisitos](#2-requisitos)
3. [Armar el entorno (una sola vez)](#3-armar-el-entorno-una-sola-vez)
4. [Uso diario del contenedor](#4-uso-diario-del-contenedor)
5. [Crear el workspace ROS 2](#5-crear-el-workspace-ros-2)
6. [Los scripts de la unidad](#6-los-scripts-de-la-unidad)
7. [Actividad 1: adquisición de datos](#7-actividad-1-adquisición-de-datos)
8. [Actividad 2: auditoría temporal](#8-actividad-2-auditoría-temporal)
9. [Cómo leer los resultados](#9-cómo-leer-los-resultados)
10. [Practicar sin ROS](#10-practicar-sin-ros)
11. [Entregables y checklist](#11-entregables-y-checklist)
12. [Resolución de problemas](#12-resolución-de-problemas)
13. [Referencia rápida de comandos](#13-referencia-rápida-de-comandos)

---

## 1. Qué vas a hacer

```
 Gazebo + TurtleBot3          generador              logger                 auditoría
 (publica /clock)   ──►  publica /cmd_vel  ──►  registra en CSV  ──►  figuras + estadísticas
                          a 10 Hz nominal        t_sec, v, w, dt        f̂, jitter, outliers
```

Al final vas a poder responder, con tus propios datos:

- ¿A qué frecuencia llegan *realmente* los comandos? (frecuencia efectiva f̂)
- ¿Cuánto varía el intervalo entre mensajes? (jitter)
- ¿Hubo huecos, pérdidas o retrasos? ¿Cuándo y por qué?

**Tiempo estimado de laboratorio:** unas 4 h 30 min. Armar el entorno por primera vez puede llevar 30–60 min más, sobre todo por la descarga de la imagen.

---

## 2. Requisitos

| Requisito | Cómo verificarlo |
|---|---|
| Linux con entorno gráfico (X11) | `echo $DISPLAY` muestra algo como `:0` o `:1` |
| Docker | `docker --version` |
| Docker Compose (v2) | `docker compose version` |
| Tu usuario puede usar Docker sin `sudo` | `docker run --rm hello-world` |
| ~10 GB libres en disco | `df -h ~` |

> Si `docker run hello-world` pide `sudo`, ejecutá `sudo usermod -aG docker $USER` y después cerrá sesión y volvé a entrar.

> **Wayland:** si tu escritorio usa Wayland (Ubuntu reciente), Gazebo puede no abrir. En ese caso, en la pantalla de inicio de sesión elegí "Ubuntu on Xorg".

---

## 3. Armar el entorno (una sola vez)

### 3.1 La carpeta del curso

La cátedra distribuye la carpeta `utec_optativa_ros2` completa. Descomprimila en tu carpeta personal; tiene que quedar así:

```
utec_optativa_ros2/
├── Dockerfile                ← cómo construir la imagen (versiones fijas)
├── docker-compose.yml        ← cómo levantar el contenedor (utec_ros2_humble)
├── entrypoint.sh
├── guia_docker_espanol.txt   ← guía paso a paso del entorno
├── codigos/
│   ├── u1/ ... u6/           ← scripts de cada unidad (esta carpeta es codigos/u1)
│   └── integrador/
└── ros2_ws/                  ← workspace de ROS 2 (lo creás en la sección 5)
```

La carpeta **completa** se monta dentro del contenedor en `/home/student/project` (`~/project`). Los scripts de esta unidad quedan en `~/project/codigos/u1`: no hay que copiar nada.

> **Regla de oro:** todo lo que quieras conservar va dentro de `~/project`. Lo que guardes en cualquier otro lugar del contenedor se pierde al hacer `docker compose down`.

### 3.2 Construir la imagen y levantar el contenedor

El `Dockerfile`, el `docker-compose.yml` y el `entrypoint.sh` están explicados línea por línea en el apunte (sección "Entorno de desarrollo") y en `guia_docker_espanol.txt`.

```bash
cd ~/utec_optativa_ros2
chmod +x entrypoint.sh
xhost +local:docker
docker compose up -d --build      # la primera vez tarda 20–40 min: descarga varios GB
```

Con versiones antiguas de Docker el comando lleva guion: `docker-compose up -d --build`.

> Si tu usuario en Linux no tiene UID 1000 (verificalo con `id -u`), construí con `docker compose build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)` y después `docker compose up -d`. Si no, puede que no tengas permiso de escritura en `~/project` desde el contenedor.

Una vez adentro, comprobá las versiones (anotalas: forman parte de la documentación de tus experimentos):

```bash
echo $ROS_DISTRO                                              # humble
python3 -c "import numpy, pandas; print(numpy.__version__, pandas.__version__)"   # 1.26.4 2.2.3
```

---

## 4. Uso diario del contenedor

```bash
cd ~/utec_optativa_ros2
xhost +local:docker                         # cada vez que reiniciás la computadora
docker compose up -d                        # levanta el contenedor en segundo plano
docker exec -it utec_ros2_humble bash       # abre una terminal adentro (repetir por cada terminal)
```

Dentro del contenedor, verificá:

```bash
echo $ROS_DISTRO        # → humble
ros2 --help | head -3   # → usage: ros2 ...
```

Al terminar la sesión:

```bash
docker compose stop     # o "docker compose down" (elimina el contenedor; ~/project se conserva)
xhost -local:docker     # opcional: revierte el permiso gráfico
```

> **Vas a necesitar 4 terminales dentro del contenedor.** Abrí cuatro pestañas en tu terminal y en cada una ejecutá `docker exec -it utec_ros2_humble bash`.

### ✅ Checkpoint 1: entorno

- [ ] `docker compose up -d` levanta el contenedor sin errores
- [ ] `ros2 launch turtlebot3_gazebo empty_world.launch.py` abre Gazebo con el robot
- [ ] En otra terminal, `ros2 topic echo /clock --once` devuelve un tiempo que **cambia** si lo repetís

---

## 5. Crear el workspace ROS 2

Dentro del contenedor:

```bash
cd ~/project/ros2_ws/src
ros2 pkg create cmdvel_tools --build-type ament_python --dependencies rclpy geometry_msgs
```

Copiá `dummy_node.py` a `~/project/ros2_ws/src/cmdvel_tools/cmdvel_tools/` y registralo en `~/project/ros2_ws/src/cmdvel_tools/setup.py`:

```python
entry_points={
    'console_scripts': [
        'dummy_node = cmdvel_tools.dummy_node:main',
    ],
},
```

Compilá y probá:

```bash
cd ~/project/ros2_ws
colcon build
source install/setup.bash
ros2 run cmdvel_tools dummy_node     # → [INFO] [dummy_node]: Dummy node running   (Ctrl+C para salir)
```

> Después de cada `colcon build`, hacé `source install/setup.bash` en las terminales que ya estaban abiertas. Las terminales nuevas lo cargan solas, gracias al `.bashrc` del Dockerfile.

### ✅ Checkpoint 2: workspace

- [ ] `colcon build` termina sin errores
- [ ] `ros2 pkg list | grep cmdvel_tools` encuentra el paquete
- [ ] `ros2 run cmdvel_tools dummy_node` imprime el mensaje esperado

---

## 6. Los scripts de la unidad

| Script | Qué hace | ¿Necesita ROS? |
|---|---|---|
| `cmd_vel_generator_basic.py` | Publica `/cmd_vel` a frecuencia fija con el ciclo avanzar → detener → girar → detener | Sí |
| `cmd_vel_logger.py` | Registra cada mensaje de `/cmd_vel` en un CSV con el tiempo de recepción | Sí |
| `quick_audit_dt.py` | Calcula f̂, jitter y outliers, y genera tres figuras | No |
| `simulate_cmdvel_csv.py` | Genera CSV sintéticos con irregularidades conocidas | No |
| `dummy_node.py` | Nodo mínimo para probar el workspace | Sí |

Todos aceptan `--help`, por ejemplo `python3 quick_audit_dt.py --help`.

### Parámetros del generador

| Parámetro | Por defecto | Significado |
|---|---|---|
| `--use-sim-time` | (apagado) | Usar el reloj de Gazebo (`/clock`). **Siempre activado en esta unidad.** |
| `--rate` | 10 | Frecuencia nominal de publicación (Hz) |
| `--v-fwd` | 0.20 | Velocidad lineal al avanzar (m/s); máximo del burger ≈ 0.22 |
| `--w-turn` | 0.80 | Velocidad angular al girar (rad/s) |
| `--t-fwd` / `--t-turn` / `--t-stop` | 6 / 3 / 2 | Duración de cada fase (s); ciclo total = 6 + 2 + 3 + 2 = 13 s |
| `--duration` | (sin límite) | Detenerse solo después de N segundos |
| `--stamped` / `--no-stamped` | según `ROS_DISTRO` | Publicar `TwistStamped` (Jazzy en adelante) o `Twist` (Humble). Normalmente no hace falta tocarlo |

### Parámetros del logger

| Parámetro | Por defecto | Significado |
|---|---|---|
| `--use-sim-time` | (apagado) | Usar el reloj de Gazebo. **Tiene que coincidir con el del generador.** |
| `--out` | `data/cmdvel_u1.csv` | Archivo de salida |
| `--duration` | 60 | Segundos de registro, contados desde el primer mensaje |
| `--stamped` / `--no-stamped` | según `ROS_DISTRO` | Esperar `TwistStamped` o `Twist`; tiene que coincidir con el generador |

### Formato del CSV

```
t_sec,v,w,dt
12.345000000,0.2,0.0,
12.445000000,0.2,0.0,0.100000000
12.546000000,0.2,0.0,0.101000000
...
```

- `t_sec`: tiempo de **recepción** según el reloj del nodo (en simulación, segundos desde que arrancó Gazebo)
- `v`, `w`: velocidad lineal (m/s) y angular (rad/s)
- `dt`: intervalo respecto de la fila anterior (vacío en la primera fila)

**Con ROS 2 Jazzy** (`/cmd_vel` es `TwistStamped`) el CSV tiene una columna más:

- `t_stamp`: hora de **emisión** (`header.stamp`), puesta por el generador

Con las dos marcas, `t_sec - t_stamp` es el retraso de cada mensaje, y podés separar el jitter del emisor del de la entrega: la descomposición ε + L del apunte. Los scripts de análisis ignoran esa columna, así que todo lo demás funciona igual.

---

## 7. Actividad 1: adquisición de datos

Seguí este orden. **El orden importa.**

**Terminal 1: simulación**
```bash
ros2 launch turtlebot3_gazebo empty_world.launch.py
```
Esperá unos 30 s, hasta que aparezca el robot. Anotá el **Real Time Factor** que muestra Gazebo en la barra inferior; lo vas a necesitar para el informe.

**Terminal 2: verificar el reloj**
```bash
ros2 topic hz /clock        # debe mostrar "average rate: ..." (Ctrl+C para salir)
ros2 topic list             # debe aparecer /cmd_vel y /clock
```
> Si `/clock` no avanza, **no sigas**: ver [Resolución de problemas](#12-resolución-de-problemas).

**Terminal 3: generador** (dejalo corriendo)
```bash
cd ~/project/codigos/u1
python3 cmd_vel_generator_basic.py --use-sim-time --rate 10 \
    --v-fwd 0.20 --w-turn 0.80 --t-fwd 6.0 --t-turn 3.0 --t-stop 2.0
```
Deberías ver el cambio de fases en la terminal y al robot moviéndose en Gazebo.

**Terminal 4: logger**
```bash
cd ~/project/codigos/u1
python3 cmd_vel_logger.py --use-sim-time --out data/cmdvel_u1.csv --duration 300
```
Termina solo a los 300 s de tiempo simulado e imprime algo como:
```
Guardadas 3001 muestras en data/cmdvel_u1.csv | duración 300.00 s | freq. efectiva 10.00 Hz
```
Si el *real time factor* es menor que 1, los 300 s simulados van a tardar más de 5 minutos reales.

**Terminal 3:** detené el generador con `Ctrl+C`. El script publica velocidad cero y el robot se detiene.

**Verificá el archivo:**
```bash
head -5 data/cmdvel_u1.csv
wc -l data/cmdvel_u1.csv      # ≈ 3000 líneas para 300 s a 10 Hz
```

> **¿Por qué 300 s?** Con un ciclo de 13 s, 300 s dan unos 23 ciclos completos: datos suficientes para estimar f̂ y jitter con confianza.

### Variante: teleoperación manual

En lugar del generador (terminal 3):
```bash
ros2 run turtlebot3_teleop teleop_keyboard     # w/x: avanzar/retroceder · a/d: girar · s: parar
```
Mové el robot **de forma continua** durante todo el registro. Si pasa más tiempo quieto que en movimiento, la señal no es representativa.

---

## 8. Actividad 2: auditoría temporal

```bash
cd ~/project/codigos/u1
python3 quick_audit_dt.py --csv data/cmdvel_u1.csv --outdir outputs_u1 --nominal-rate 10
```

Archivos que genera en `outputs_u1/`:

| Archivo | Contenido |
|---|---|
| `velocity_signals.png` | v(t) y ω(t): ¿se ve el patrón del generador? |
| `dt_series.png` | Δt a lo largo del tiempo, con la mediana, el umbral 3 × mediana y la banda de Hampel |
| `dt_histogram.png` | Distribución de Δt |
| `summary_stats.txt` | Todas las estadísticas y la lista de intervalos atípicos |
| `dt_outliers.csv` | Intervalos marcados (solo si hay) |

Opciones útiles:

```bash
--log-hist          # eje y logarítmico en el histograma: hace visibles las colas
--k-ratio 2         # cambia el umbral de la regla de razón (por defecto 3)
--hampel-c 2.5      # cambia la constante de Hampel (por defecto 3)
```

> Probá distintos umbrales y fijate cómo cambia la cantidad de outliers. Justificar **qué umbral elegiste y por qué** es parte de la evaluación.

---

## 9. Cómo leer los resultados

### `summary_stats.txt`, línea por línea

| Línea | Qué te dice | Qué esperar en simulación a 10 Hz |
|---|---|---|
| `Freq. efectiva` | 1 / mediana(Δt): la frecuencia real | ≈ 10 Hz |
| `Freq. con la media` | La misma cuenta con la media: **no la uses**, está para comparar | Menor que f̂ si hubo pausas |
| `Std(dt)` vs. `1.4826*MAD` | Si la std es mucho mayor, unos pocos valores extremos la inflan | Parecidas, si no hubo eventos |
| `P99(dt)` | El 1 % de los intervalos dura al menos esto | Cerca de la mediana |
| `Regla de razón` | Huecos de 3 períodos o más (pausas, cortes) | 0 o muy pocos |
| `Hampel` | Cualquier intervalo fuera de la variabilidad típica | Algunos |
| `dt = 0` | Dos mensajes en el mismo paso de simulación | Posible en simulación; no es un error |
| `dt < 0` | **El tiempo retrocedió** (se reinició Gazebo) | **Debe ser 0.** Si no, repetí el registro. |

### Patrones típicos en las figuras

| Lo que ves | Lo que probablemente pasó |
|---|---|
| Histograma angosto, centrado en 0.100 s | Muestreo estable ✅ |
| Un segundo pico en ≈ 0.200 s (2T) | **Mensajes perdidos**: falta uno entre dos que llegaron |
| Pico cerca de 0 s | Mensajes duplicados o en ráfaga |
| En la serie, un Δt largo seguido de uno corto | Un mensaje **retrasado** (no perdido): el siguiente "recupera" el plan |
| Un único Δt enorme (segundos) | Pausa: se pausó Gazebo, el generador o hubo un bloqueo |
| Δt "en escalones" de 1 ms | Cuantización del reloj simulado: es normal con `use_sim_time` |
| v(t) con valores imposibles (> 0.22 m/s) | Error en el registro o en el generador |

> **Ojo con el lenguaje:** Δt **no mide latencia**. Si todos los mensajes llegaran con 50 ms de retraso constante, Δt sería igual. Δt solo ve las *variaciones* (ver "Un modelo del instante de llegada" en el apunte).

---

## 10. Practicar sin ROS

`simulate_cmdvel_csv.py` genera CSV con el mismo formato que el logger, con irregularidades que vos elegís. Sirve para entender las figuras **antes** de tener datos reales, o para practicar en tu casa sin Docker:

```bash
# flujo limpio
python3 simulate_cmdvel_csv.py --out data/sint_limpio.csv

# con eventos: 1 % de pérdidas, pausa de 2 s en t = 150 s, 5 retrasos aislados,
# reloj cuantizado a 1 ms (como en simulación)
python3 simulate_cmdvel_csv.py --out data/sint_eventos.csv \
    --drop-prob 0.01 --pause-at 150 --pause-len 2.0 --n-delays 5 --quantum 0.001

python3 quick_audit_dt.py --csv data/sint_eventos.csv --outdir outputs_sint --nominal-rate 10
```

**Ejercicio:** en las figuras de `outputs_sint/`, encontrá la pausa, las pérdidas y los retrasos. ¿Qué criterio de outliers detecta cada uno?

Fuera del contenedor solo necesitás `pip install "numpy<2" pandas matplotlib` (los scripts de análisis no usan ROS).

---

## 11. Entregables y checklist

### Entregable A1: datos

- `cmdvel_u1.csv`
- Texto breve (5–8 líneas) con:
  1. duración del experimento y cantidad de muestras
  2. frecuencia nominal
  3. si usaste `use_sim_time` y qué *real time factor* mostró Gazebo
  4. el patrón de comandos (o descripción de la teleoperación)

*Ejemplo:* "Se registraron 3001 muestras durante 300 s con el generador automático, con tiempo de simulación (`use_sim_time=true`, RTF ≈ 0.98). Frecuencia nominal: 10 Hz. Patrón: avanzar 6 s a 0.20 m/s, detenerse 2 s, girar 3 s a 0.80 rad/s, detenerse 2 s; ≈ 23 ciclos."

### Entregable A2: análisis

- Carpeta `outputs_u1/` completa
- Párrafo técnico (8–12 líneas) con:
  1. frecuencia efectiva f̂ comparada con la nominal
  2. jitter (std **y** MAD), y qué significa que sean parecidos o distintos
  3. presencia o ausencia de irregularidades, **con el criterio y el umbral usados**
  4. interpretación de los outliers: cuándo ocurren y una causa plausible

### ✅ Checklist antes de entregar

- [ ] El CSV tiene tiempos monótonos (`dt < 0: 0` en el resumen)
- [ ] Reporté f̂ calculada con la **mediana**
- [ ] Reporté el jitter y dije qué estimador usé
- [ ] Las figuras se leen bien (ejes con unidades, títulos)
- [ ] Justifiqué el criterio de outliers con **mis** datos, no de forma genérica
- [ ] Documenté los parámetros: frecuencia, duración, reloj, RTF, versión de ROS 2 (Humble)

### Qué se evalúa

| Criterio | Insuficiente | Aceptable | Destacado |
|---|---|---|---|
| Comprensión conceptual | Confunde tiempo de sistema y de simulación, o Δt con frecuencia | Aplica bien los conceptos sin explicarlos | Explica *por qué* cada métrica se calcula así |
| Justificación técnica | Reporta números sin explicar de dónde salen | Justifica mediana/MAD de forma genérica | Justifica cada elección con el caso concreto de sus datos |
| Análisis de outliers | No los identifica o los ignora | Los identifica y reporta | Los identifica, propone una causa y discute si afectan la interpretación |

---

## 12. Resolución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ros2: command not found` | Estás fuera del contenedor, o la imagen se construyó sin la línea del `.bashrc` | `docker exec -it utec_ros2_humble bash`; si persiste, `source /opt/ros/humble/setup.bash` y reconstruí la imagen |
| Gazebo no abre / `cannot open display` | Falta el permiso gráfico | En el host: `xhost +local:docker`; verificá `echo $DISPLAY` |
| Gazebo abre en negro o se cierra | Driver de video o Wayland | Iniciá sesión con "Ubuntu on Xorg"; esperá un poco más la primera vez |
| El logger dice "Mensaje descartado: /clock todavía no publicó tiempo" | Gazebo no está corriendo o no terminó de cargar | Esperá; verificá con `ros2 topic hz /clock` |
| El logger no registra nada | El generador no está publicando, o los relojes no coinciden | `ros2 topic hz /cmd_vel`; usá `--use-sim-time` en **ambos** |
| Los `t_sec` son ≈ 1.7×10⁹ | El logger usó el reloj del sistema | Te faltó `--use-sim-time` en el logger |
| Aparecen tópicos que no lanzaste | Compartís el dominio DDS con otras máquinas | Verificá `ROS_LOCALHOST_ONLY=1` en `docker-compose.yml` |
| Hay el doble de mensajes de los esperados | Dos generadores corriendo, propios o ajenos | `ros2 node list`; cerrá los que sobran |
| `dt < 0` en el resumen | Se reinició la simulación durante el registro | Repetí el registro sin tocar Gazebo |
| `Permission denied` al guardar en `~/project` | Tu UID no es 1000 | Reconstruí con `--build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)` |
| Error de numpy al importar algo de ROS | numpy 2.x instalado | Reconstruí la imagen con el Dockerfile de esta guía (`"numpy<2"`) |
| Los cambios desaparecen | Trabajaste fuera de `~/project` | Guardá siempre dentro de `~/project` |
| En Jazzy el robot no se mueve, o `ros2 topic info /cmd_vel` muestra dos tipos | El generador publica `Twist` y TurtleBot3 espera `TwistStamped` | Verificá `echo $ROS_DISTRO`; forzá `--stamped` en el generador y en el logger |
| El robot sigue moviéndose después de cortar el generador | Se cerró la terminal sin Ctrl+C | `ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"` |

Si nada de esto lo resuelve, antes de consultar anotá **el comando exacto** que ejecutaste y **el mensaje de error completo**.

---

## 13. Referencia rápida de comandos

```bash
# ── Host ────────────────────────────────────────────
xhost +local:docker
docker compose up -d
docker exec -it utec_ros2_humble bash
docker compose down

# ── Contenedor: diagnóstico ─────────────────────────
ros2 topic list
ros2 topic hz /clock
ros2 topic hz /cmd_vel
ros2 topic echo /cmd_vel
ros2 node list

# ── Contenedor: experimento ─────────────────────────
ros2 launch turtlebot3_gazebo empty_world.launch.py
python3 cmd_vel_generator_basic.py --use-sim-time --rate 10
python3 cmd_vel_logger.py --use-sim-time --out data/cmdvel_u1.csv --duration 300
python3 quick_audit_dt.py --csv data/cmdvel_u1.csv --outdir outputs_u1 --nominal-rate 10

# ── Workspace ───────────────────────────────────────
cd ~/project/ros2_ws && colcon build && source install/setup.bash
```

---

### Lecturas para profundizar

- Documentación de ROS 2 sobre el tiempo: <https://docs.ros.org/en/humble/Concepts/About-ROS-2-Time.html>
- Simulación del TurtleBot3: <https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/>
- La bibliografía completa, organizada por tema, está al final del apunte de la unidad.
