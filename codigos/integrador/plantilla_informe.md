# Informe de auditoría: registro de /cmd_vel

**Grupo:** …  **Fecha:** …  **Caso:** …

## 1. Resumen ejecutivo (media página)

Qué pasó, cuándo y con qué confianza, en lenguaje para el equipo de operación.

## 2. Datos y método

- **Registros:** duración, período efectivo y jitter de cada uno.
- **Grilla y ventanas:** decisión y justificación (Unidad 2).
- **Indicadores:** cuáles, con qué parámetros y por qué (Unidades 3 y 4).
- **Calibración del detector:** registro usado, normalización, pesos, τ, K o CUSUM (Unidad 5).
- **Modelos:** tipo, T, L, canales, si aportan o no y con qué evidencia (Unidad 6).

## 3. Hallazgos

| # | Inicio (s) | Fin (s) | Indicadores que lo muestran | Indicador principal | Retardo de detección (s) |
|---|---|---|---|---|---|
| 1 | | | | | |

Figura: hallazgos ubicados en el tiempo, sobre el score y los indicadores.

**Otras alarmas:**

| Intervalo (s) | Clasificación (cola de ventana / hallazgo menor / falsa alarma) | Justificación |
|---|---|---|
| | | |

## 4. Diagnóstico

Escala de confianza usada: …

| # | Síntoma | Evidencia | Causas compatibles | Causas descartadas (y por qué) | Confianza |
|---|---|---|---|---|---|
| 1 | | | | | |

## 5. Límites

Qué no pudo ver el análisis y qué información adicional cambiaría las conclusiones: otros tópicos, `header.stamp`, cantidad de publicadores, registros del sistema…

## 6. Anexo de reproducibilidad

- Versiones: Python, numpy, pandas, PyTorch e imagen Docker.
- Semillas.
- Comandos, en orden, que regeneran cada tabla y cada figura:

```bash
# …
```
