# Parte III — Optimizador v2 (Cambios de calibre con setup time) + extensiones

## 1) Objetivo del optimizador v2
Permitir que una salida cambie de calibre durante el turno, pagando un **setup time** (p.ej. 20 minutos), para adaptarse a:

- cambios de mix entre fincas,
- periodos donde conviene reasignar capacidad,
- estrategias de “re-balanceo” en transición finca→finca.

Este v2 se modela mejor como un **problema de scheduling**.

---

## 2) Datos adicionales requeridos
- Setup time `S` (minutos) por cambio de calibre en una salida.
- (Opcional) Ventanas de tiempo por finca: si las fincas entran secuencialmente y cada una dura un tiempo asociado al flujo/entrada, el plan debe respetar esa secuencia.

Hay dos enfoques:

### Enfoque A — Scheduling “por ventanas” (recomendado)
Dividir el turno en segmentos (p.ej. por finca o por bloques fijos de tiempo). El optimizador decide qué calibre procesa cada salida en cada segmento, con setups entre segmentos.

### Enfoque B — Scheduling continuo con intervalos variables
El optimizador decide tiempos exactos de cambio (más potente, más complejo).

---

## 3) Formulación por bloques (simple y efectiva)

### 3.1 Indexación por bloques
Sea `K` el número de bloques por salida (p.ej. 1 bloque por finca, o 8 bloques de 1 hora).

Variables:
- \(y_{m,c,k} \in \{0,1\}\): salida `m` procesa calibre `c` en bloque `k`.
- \(setup_{m,k} \in \{0,1\}\): hay cambio de calibre entre bloque `k-1` y `k` en salida `m`.

Restricciones:
- En cada bloque, cada salida elige un calibre:
  \[
  \sum_c y_{m,c,k} = 1
  \]
- Definir `setup`:
  \[
  setup_{m,k} \ge y_{m,c,k} - y_{m,c,k-1} \quad \forall c
  \]
  (hay variantes más ajustadas; esta es una simplificación)

Capacidad efectiva en el bloque:
- Duración bloque: `T_k` horas
- Setup consume `S` horas si `setup_{m,k}=1`
- Tiempo productivo: `T_k - S * setup_{m,k}`

Producción de calibre `c` en bloque `k`:
\[
Prod_{c,k} = \sum_m v_m \, y_{m,c,k}\, (T_k - S \cdot setup_{m,k})
\]

Demanda por bloque:
- Si segmentás por finca, cada bloque `k` representa una finca `f(k)` y demanda por calibre es `K[f,c]`.
- Si segmentás por tiempo fijo, la demanda se reparte proporcionalmente.

Objetivo:
- Maximizar producción total dentro del turno, o minimizar faltantes:
  \[
  \max \sum_{c,k} Prod_{c,k} - \alpha \sum_{c,k} Short_{c,k}
  \]
o
  \[
  \min \sum_{c} Short_c + \beta \sum_{m,k} setup_{m,k}
  \]

---

## 4) Por qué OR-Tools CP-SAT
Aunque lo anterior puede implementarse como MILP, CP-SAT suele ser más práctico cuando:

- hay muchas binarias,
- se agregan restricciones lógicas,
- se agregan ventanas/intervalos,
- querés extender a reordenar fincas (se vuelve más “combinatorio”).

---

## 5) Skeleton OR-Tools CP-SAT (bloques discretos)
Este skeleton asume bloques `k=0..K-1` con duración fija `T_k` y setup time fijo `S`.

```python
from ortools.sat.python import cp_model

def solve_v2_blocks_cpsat(
    v_m,                # list length M
    calibers,           # list of calibers
    T_hours,            # list length K
    setup_hours,        # float (e.g. 20/60)
    demand_by_block,    # list length K of dict {caliber: kg demand} (optional)
    setup_penalty=1.0,
    short_penalty=1000.0,
):
    M = len(v_m)
    C = len(calibers)
    K = len(T_hours)

    model = cp_model.CpModel()

    y = {}
    for m in range(M):
        for k in range(K):
            for c in calibers:
                y[(m,c,k)] = model.NewBoolVar(f"y_m{m}_c{c}_k{k}")

    setup = {}
    for m in range(M):
        for k in range(1, K):
            setup[(m,k)] = model.NewBoolVar(f"setup_m{m}_k{k}")

    # each outlet picks exactly 1 caliber per block
    for m in range(M):
        for k in range(K):
            model.Add(sum(y[(m,c,k)] for c in calibers) == 1)

    # setup detection (simplified)
    for m in range(M):
        for k in range(1, K):
            for c in calibers:
                model.Add(setup[(m,k)] >= y[(m,c,k)] - y[(m,c,k-1)])

    # production and shortages
    short = {}
    prod = {}
    # Use integer scaling to avoid floats in CP-SAT:
    SCALE = 1000
    v_int = [int(round(vm * SCALE)) for vm in v_m]
    T_int = [int(round(t * SCALE)) for t in T_hours]
    S_int = int(round(setup_hours * SCALE))

    for k in range(K):
        for c in calibers:
            # prod_{c,k} = sum_m v_m * y * (T_k - S*setup)
            # We'll create an IntVar upper-bounded conservatively.
            ub = sum(v_int[m] * T_int[k] for m in range(M))
            prod[(c,k)] = model.NewIntVar(0, ub, f"prod_c{c}_k{k}")

            # Build expression
            terms = []
            for m in range(M):
                # productive time depends on setup at (m,k), for k>=1
                if k == 0:
                    time_int = T_int[k]
                    terms.append(v_int[m] * time_int * y[(m,c,k)])
                else:
                    # time = T - S*setup
                    # Linearization: v*T*y - v*S*y*setup is not linear.
                    # So for CP-SAT blocks, a common simplification is:
                    # treat setup as capacity loss per outlet per block regardless of chosen caliber,
                    # and subtract v*S*setup from total outlet capacity separately.
                    terms.append(v_int[m] * T_int[k] * y[(m,c,k)])

            # Sum of terms
            model.Add(prod[(c,k)] == sum(terms))

            # shortage if demand exists
            d = int(round(demand_by_block[k].get(c, 0.0) * SCALE * SCALE))
            short[(c,k)] = model.NewIntVar(0, d, f"short_c{c}_k{k}")
            # prod + short >= demand (after scaling alignment)
            model.Add(prod[(c,k)] + short[(c,k)] >= d)

    # setup penalty (capacity loss) — handled in objective rather than exact prod
    setup_cost = sum(setup[(m,k)] for m in range(M) for k in range(1,K))

    # objective: minimize shortages + penalize setups
    model.Minimize(
        short_penalty * sum(short.values()) + int(setup_penalty) * setup_cost
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)

    plan = [[None]*K for _ in range(M)]
    for m in range(M):
        for k in range(K):
            for c in calibers:
                if solver.Value(y[(m,c,k)]) == 1:
                    plan[m][k] = c
                    break

    return {
        "status": status,
        "plan_caliber_by_outlet_by_block": plan,
        "setup_count": sum(solver.Value(setup[(m,k)]) for m in range(M) for k in range(1,K)) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
    }
```

### Importante sobre el skeleton
- El producto `y * setup` introduce bilinearidad. En v2 real, se modela de dos maneras:
  1) **Bloques + setup como pérdida de tiempo fija por cambio**, pero imputada como penalización (aprox).
  2) Modelo más exacto con variables auxiliares `z = y AND setup` por calibre, lo cual es linealizable en CP-SAT.

Si querés exactitud, te conviene:
- modelar “intervalos” de setup y producción por salida (más detallado),
- o linearizar con `z_{m,c,k} ≤ y_{m,c,k}`, `z_{m,c,k} ≤ setup_{m,k}`, `z ≥ y+setup-1`.

---

## 6) Integración con simulación
Una vez obtenido un plan por bloques:
- Convertirlo a una política en el simulador:
  - en el inicio de cada bloque, actualizar `assignment.caliber_by_outlet[m]` según el plan,
  - simular un “setup downtime” durante `S` minutos para cada salida que cambia (la salida no empaqueta durante ese lapso).

Esto permite comparar v1 vs v2 bajo el mismo generador estocástico.

---

## 7) Extensiones
### 7.1 Buffer sizing
- Correr simulación variando `B` (kg) y medir throughput marginal.
- Identificar punto de rendimientos decrecientes.

### 7.2 Reordenamiento de fincas
- Convertir el orden de fincas en variable de decisión (TSP-like / scheduling).
- Heurística sugerida: ordenar para minimizar variación de `p_{f,c}` entre fincas consecutivas (reduce setups y desbalance).

### 7.3 Setup dependiente de calibre (si aplica)
- `S(c_old, c_new)` en lugar de setup fijo.

---

## 8) Checklist DoD — v2
- [ ] Genera plan por bloques con setups contados.
- [ ] Simulador soporta cambios de calibre y downtime.
- [ ] Reporte comparativo v1 vs v2: throughput, bloqueo, setups, utilización.
- [ ] Experimentos con distintos tamaños de buffer.
