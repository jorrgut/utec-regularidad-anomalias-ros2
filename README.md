# Análisis de regularidad y detección de anomalías en sistemas robóticos

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24425122.svg)](https://doi.org/10.5281/zenodo.24425122)
[![smoke-test](https://github.com/jorrgut/utec-regularidad-anomalias-ros2/actions/workflows/smoke.yml/badge.svg)](https://github.com/jorrgut/utec-regularidad-anomalias-ros2/actions/workflows/smoke.yml)

Códigos y entorno de laboratorio de la optativa *Análisis de Regularidad y Detección de Anomalías en Sistemas Robóticos*, Ingeniería en Mecatrónica, Universidad Tecnológica del Uruguay (UTEC).

El libro del curso (*Introducción al análisis de regularidad y a la detección de anomalías en sistemas robóticos*) explica la teoría y las actividades; este repositorio tiene todo lo que se ejecuta: los scripts de las seis unidades, el proyecto integrador y dos entornos Docker reproducibles.

| | Entorno principal | Entorno alternativo |
|---|---|---|
| Sistema | Ubuntu 22.04 | Ubuntu 24.04 |
| ROS 2 | Humble (soporte hasta mayo de 2027) | Jazzy (soporte hasta mayo de 2029) |
| Simulador | Gazebo Classic 11 | Gazebo Harmonic |
| Python | 3.10 | 3.12 |
| `/cmd_vel` | `geometry_msgs/Twist` | `geometry_msgs/TwistStamped` |
| Carpeta | raíz del repositorio | `jazzy/` |

Las versiones de Python están fijas en los dos entornos: numpy 1.26.4, pandas 2.2.3, matplotlib 3.8.4 y PyTorch 2.5.1 (CPU). Los scripts y los nodos funcionan sin cambios en los dos: detectan la distribución de ROS 2 y usan el tipo de mensaje que corresponde.

---

## Estructura

```
.
├── Dockerfile, docker-compose.yml, entrypoint.sh   ← entorno Humble (el del libro)
├── guia_docker_espanol.txt, guia_docker_portugues.txt
├── jazzy/                     ← entorno Jazzy (Dockerfile, docker-compose.yml, guías)
├── codigos/
│   ├── u1/ … u6/              ← scripts de cada unidad, cada una con su README
│   └── integrador/            ← generador de casos del proyecto integrador
├── ros2_ws/                   ← workspace de ROS 2 (Humble)
├── ros2_ws_jazzy/             ← workspace de ROS 2 (Jazzy)
└── tests/smoke_test.sh        ← prueba rápida de todos los scripts, sin ROS
```

El repositorio completo se monta dentro del contenedor en `~/project`. Los scripts de la unidad *N* se ejecutan desde `~/project/codigos/uN`.

---

## Empezar

Requisitos: Linux con entorno gráfico (X11), Docker y Docker Compose, y unos 10 GB libres.

```bash
git clone https://github.com/jorrgut/utec-regularidad-anomalias-ros2.git
cd utec-regularidad-anomalias-ros2
xhost +local:docker
docker compose up -d --build              # la primera vez tarda 20–40 min
docker exec -it utec_ros2_humble bash     # una terminal por cada proceso
```

Dentro del contenedor:

```bash
echo $ROS_DISTRO                           # humble
cd ~/project/codigos/u1
```

y seguí la Unidad 1 del libro (o `codigos/u1/README.md`).

**Entorno Jazzy:** los mismos pasos, desde la carpeta `jazzy/`:

```bash
cd jazzy
docker compose up -d --build
docker exec -it utec_ros2_jazzy bash
```

En Jazzy el workspace de ROS 2 es `ros2_ws_jazzy/` del repositorio, que dentro del contenedor aparece igual como `~/project/ros2_ws`. Así lo compilado para una distribución no interfiere con la otra.

**Sin Git:** la cátedra también distribuye cada entorno como un archivo comprimido (`utec_optativa_ros2` y `utec_optativa_ros2_jazzy`), con los mismos códigos.

---

## Probar los scripts sin ROS

Los scripts de análisis no necesitan ROS 2. Para comprobar que todo funciona (en el contenedor o en cualquier Python 3.10–3.12 con las versiones de arriba):

```bash
bash tests/smoke_test.sh
```

La prueba genera datos sintéticos y recorre el flujo de las seis unidades y del proyecto integrador en una copia temporal; no escribe nada en el repositorio. Es la misma que corre la integración continua en cada cambio.

---

## Versiones

Cada edición del curso corresponde a una versión etiquetada (`v2026.1`, …). Para reproducir los resultados de una edición, usá su etiqueta:

```bash
git checkout v2026.1
```

---

## Cómo citar

Si usás estos materiales, citá la versión que usaste. Los datos están en [`CITATION.cff`](CITATION.cff) (GitHub ofrece el formato APA y BibTeX en "Cite this repository").

> Gutiérrez, J. N. (2026). *Análisis de regularidad y detección de anomalías en sistemas robóticos: códigos y entorno de laboratorio* (versión 2026.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.24425122

## Licencia

- **Código** (archivos `.py`, `.sh`, `Dockerfile`, `docker-compose.yml`): [MIT](LICENSE).
- **Textos** (README, guías y plantillas): [CC BY 4.0](LICENSE-docs.md).

© 2026 Jorge N. Gutiérrez, Universidad Tecnológica del Uruguay.
