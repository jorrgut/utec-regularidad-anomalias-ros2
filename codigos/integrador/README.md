# Proyecto integrador: auditoría de un robot

Guía práctica del proyecto integrador (capítulo final del libro). Durante **4 semanas**, en grupos de 2 o 3, auditan el registro de `/cmd_vel` de un robot que "en algún momento se comportó raro", usando las herramientas de las **seis unidades**.

## 1. Qué reciben

| Archivo | Contenido |
|---|---|
| `registro_auditoria.csv` | 15 minutos a auditar (`t_sec, v, w`) |
| `nominal_entrenamiento.csv` | 10 minutos en operación normal (para entrenar modelos, Unidad 6) |
| `nominal_calibracion.csv` | 5 minutos en operación normal, otra corrida (para calibrar el detector, Unidad 5) |
| `ficha_robot.txt` | lo que se sabe del sistema |

**Regla de oro:** todo lo que se ajusta (grilla, ventanas, pesos, τ, K, modelos) se ajusta con los registros **nominales**. El registro auditado se analiza al final, una sola vez.

## 2. Practicar primero con el caso resuelto

```bash
cd ~/project/codigos/integrador
python3 generar_caso.py --practica --salida caso_practica
```

Genera los mismos cuatro archivos y, además, `eventos_practica.txt` con los cinco eventos del caso. El libro resuelve este caso en la sección "Un caso de práctica resuelto": úsenlo para probar su flujo de trabajo de punta a punta antes de recibir el caso evaluado.

Un recorrido mínimo con los scripts de las unidades:

```bash
P=caso_practica
# Semana 1: tiempos (Unidad 1)
python3 ../u1/quick_audit_dt.py --csv $P/registro_auditoria.csv --outdir $P/u1_auditoria
python3 ../u1/quick_audit_dt.py --csv $P/nominal_calibracion.csv --outdir $P/u1_nominal
# Semana 3: modelos (Unidad 6) y detector combinado (Unidades 5 y 6)
cd ../u6
python3 train_model.py --csv ../integrador/$P/nominal_entrenamiento.csv --kind ae --T 40 --counts \
    --out ../integrador/$P/models/ae_T40_nk
python3 u6_activity4.py --models ../integrador/$P/models/ae_T40_nk \
    --calib-csv ../integrador/$P/nominal_calibracion.csv \
    --events-csv ../integrador/$P/registro_auditoria.csv \
    --events ../integrador/$P/eventos_practica.txt --out-dir ../integrador/$P/detector
```

Con el caso evaluado **no hay archivo de eventos**: corran `u6_activity4.py` con `--events ""`. En lugar de evaluar por eventos, el script lista los intervalos de alarma de cada configuración, con el indicador que más aporta en cada uno (`act4_alarmas.csv`). Esa lista es el punto de partida del diagnóstico.

## 3. Cronograma y entregas

| Semana | Unidades | Trabajo | Entrega parcial |
|---|---|---|---|
| 1 | 1 y 2 | Auditar Δt en los tres registros; re-muestrear; elegir ventanas | Tabla de irregularidades temporales + grilla y ventanas justificadas |
| 2 | 3 y 4 | Entropías y predictibilidad; rangos nominales | Perfil de regularidad con figura alineada en el tiempo |
| 3 | 5 y 6 | Calibrar el detector; entrenar y evaluar un modelo, con y sin canal `n_k` | Calibración + lista de intervalos de alarma |
| 4 | todas | Clasificar alarmas; diagnóstico; informe | **Informe de auditoría + código** |

Las entregas parciales tienen como máximo 2 páginas, más el código.

## 4. El informe

Usen `plantilla_informe.md`, que tiene la estructura pedida: resumen ejecutivo, datos y método, hallazgos, diagnóstico, límites y anexo de reproducibilidad. Para cada alarma, decidan si es:

- un **hallazgo**;
- una **consecuencia** de otro hallazgo (por ejemplo, la cola de una ventana de 5 s);
- un **hallazgo menor**, como un retraso aislado real;
- una **falsa alarma** estadística.

En el diagnóstico, cada hallazgo lleva sus causas compatibles, las descartadas (con el porqué) y una confianza en una escala que ustedes definen. La tabla "De los síntomas a las causas compatibles" del libro es el punto de partida.

## 5. Errores frecuentes

- **Calibrar o elegir τ mirando el registro auditado.** Es fuga de información, y todo el análisis queda invalidado.
- **Atribuir causas a partir del indicador** ("H(Δt) subió, entonces fue un ataque"). El indicador dice qué propiedad cambió, no por qué.
- **Contar la cola de las ventanas como hallazgos nuevos.**
- **Dejar resultados que no se pueden regenerar.** El anexo de reproducibilidad tiene que reconstruir cada tabla y cada figura.
