# Parte II — Optimizador v1 (Asignación estática de salidas a calibres)

## 1) Objetivo del optimizador v1
Encontrar una asignación estática `caliber_by_outlet` (una vez por turno) que:

- balancee la capacidad por calibre frente al mix de entrada,
- sea **robusta** a cambios de composición entre fincas,
- maximice una tasa de producción sostenible (kg/h),
- luego se valide en el simulador (Parte I).

Este v1 **no** permite cambios de calibre durante el turno.

---

## 2) Formulación matemática (robusta por finca)

### Parámetros
- Salidas \(m \in \mathcal{M}\), calibres \(c \in \mathcal{C}\), fincas \(f \in \mathcal{F}\)
- Velocidad por salida: \(v_m\) (kg/h)
- Proporción por finca y calibre: \(p_{f,c}\) (derivada de preconteo)
  \[
  p_{f,c} = \frac{K_{f,c}}{K_f},\quad K_f=\sum_c K_{f,c}
  \]

### Variables de decisión
- \(x_{m,c} \in \{0,1\}\): 1 si salida `m` asignada a calibre `c`

Restricción de asignación:
\[
\sum_{c} x_{m,c} = 1 \quad \forall m
\]

Capacidad total por calibre:
\[
V_c = \sum_m v_m x_{m,c}
\]

### Variable objetivo
- \(\lambda \ge 0\): tasa total (kg/h) sostenible para todas las fincas.

Restricciones de robustez por finca:
\[
V_c \ge \lambda \, p_{f,c} \quad \forall f,c
\]

Objetivo:
\[
\max \lambda
\]

**Interpretación:** buscamos la mayor tasa total \(\lambda\) tal que para cualquier finca `f`, la capacidad asignada a cada calibre `c` alcance su “parte” del mix.

---

## 3) Extensiones simples (opcionales en v1)
Dependiendo de la operación, pueden agregarse:

### 3.1 Calibres obligatorios / mínimos
Si cierto calibre siempre debe tener al menos una salida:
\[
\sum_m x_{m,c} \ge 1
\]

### 3.2 Penalizar cambios de asignación entre días (si hay baseline)
Si hay asignación previa \(x^0_{m,c}\), agregar penalidad por distancia (esto ya es MILP con término en objetivo).

### 3.3 Robustez “suavizada” (si no querés min-max duro)
En lugar de max-min, usar:
- maximizar throughput esperado ponderando fincas por volumen,
- o restricciones por percentiles de fincas.

---

## 4) Implementación en Python (PuLP)

### 4.1 Estructura de inputs
- `outlet_types`: lista len M
- `speed_kgph`: dict por tipo
- `farms`: lista de `FarmLot` con `kg_by_caliber`

Derivar:
- set de calibres
- `v_m`
- `p[f,c]`

### 4.2 Skeleton con PuLP
```python
from dataclasses import dataclass
from typing import Dict, List
import pulp

def solve_assignment_v1_pulp(
    outlet_types: List[str],
    speed_kgph: Dict[str, float],
    farms_kg_by_caliber: List[Dict[int, float]],  # one dict per farm
):
    M = len(outlet_types)
    calibers = sorted({c for farm in farms_kg_by_caliber for c in farm.keys()})

    v_m = [speed_kgph[t] for t in outlet_types]

    # proportions p[f,c]
    p = []
    for farm in farms_kg_by_caliber:
        total = sum(farm.values())
        p.append({c: (farm.get(c, 0.0) / total if total > 0 else 0.0) for c in calibers})

    model = pulp.LpProblem("packing_assignment_v1", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", (range(M), calibers), lowBound=0, upBound=1, cat="Binary")
    lam = pulp.LpVariable("lambda", lowBound=0, cat="Continuous")

    # objective
    model += lam

    # each outlet assigned to exactly one caliber
    for m in range(M):
        model += pulp.lpSum(x[m][c] for c in calibers) == 1

    # robust constraints: V_c >= lam * p[f,c]
    for f in range(len(p)):
        for c in calibers:
            V_c = pulp.lpSum(v_m[m] * x[m][c] for m in range(M))
            model += V_c >= lam * p[f][c]

    # solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    assignment = []
    for m in range(M):
        chosen = max(calibers, key=lambda c: pulp.value(x[m][c]))
        assignment.append(chosen)

    return {
        "lambda_star": pulp.value(lam),
        "caliber_by_outlet": assignment,
        "status": pulp.LpStatus[model.status],
    }
```

---

## 5) Validación con el simulador
Pipeline recomendado:

1. Resolver v1 ⇒ obtener `caliber_by_outlet` y `lambda_star`.
2. Correr simulación con esa asignación, N seeds.
3. Reportar:
   - promedio y P10/P90 de throughput,
   - bloqueo, colas máximas,
   - utilización.

### Ejemplo de loop
```python
import numpy as np
import pandas as pd

def validate_assignment(
    config, farms, assignment, arrival_rate_kgph, seeds=range(50)
):
    rows = []
    for s in seeds:
        res = run_simulation(config, farms, assignment, arrival_rate_kgph, seed=s)
        rows.append({
            "seed": s,
            "total_packed_kg": res.total_packed_kg,
            "blocked_h": res.blocked_time_infeed_hours,
        })
    df = pd.DataFrame(rows)
    return df.describe(percentiles=[0.1, 0.5, 0.9])
```

**Nota práctica:** para estresar el sistema y que el cuello sea el empaque, setear `arrival_rate_kgph` grande (ej. 5–10× sum(v_m)) y dejar que el bloqueo determine el throughput efectivo.

---

## 6) Checklist DoD — v1
- [ ] Resuelve en segundos para M<=12 y C<=10.
- [ ] Devuelve asignación válida (una salida → un calibre).
- [ ] `lambda_star` razonable (no infinita, no 0 salvo casos patológicos).
- [ ] Validación por simulación muestra mejoras vs baseline (si existe).
