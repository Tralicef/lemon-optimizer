# Parte I — Simulador (Discrete-Event Simulation) con SimPy

## 1) Objetivo del simulador
Construir un **simulador de eventos discretos** (DES) que reproduzca el comportamiento operativo del empaque bajo:

- mezcla estocástica de calibres dentro de cada finca,
- buffers pequeños (bloqueo/estrangulamiento),
- múltiples salidas con distintas velocidades,
- configuración (asignación) de calibre por salida (estática o dinámica),
- turno de duración `H` (paramétrico).

El simulador se usa para:
- evaluar configuraciones propuestas por el optimizador,
- medir sensibilidad a la variabilidad del mix,
- dimensionar buffers (qué mejora da aumentar almacenamiento),
- estimar métricas operativas (utilización, bloqueo, colas).

---

## 2) Definición formal del sistema

### Conjuntos
- Fincas: \( f \in \mathcal{F} \)
- Calibres: \( c \in \mathcal{C} \) (ej. {80, 100, 120})
- Salidas: \( m \in \mathcal{M} \) (ej. 8 salidas)

### Parámetros
- \(K_{f,c}\): kg totales del calibre `c` que ingresan durante la finca `f`. **Conocido por preconteo.**
- \(K_f=\sum_c K_{f,c}\): kg totales de la finca `f`.
- \(p_{f,c}=K_{f,c}/K_f\): proporción por calibre dentro de finca `f`.
- Tipos de salida: `type(m) ∈ {AUTO, BULK, MANUAL}` (tipo fijo por salida).
- Velocidades: `v_AUTO`, `v_BULK`, `v_MANUAL` (kg/h).
- \(v_m = v_{type(m)}\): velocidad de la salida `m`.
- Turno: `H` horas (paramétrico; default 8h).
- Granularidad de llegada: `dt` (segundos) o micro-lote (kg).
- Buffer: `B_c` kg máximo para calibre `c` (opcional). Alternativamente un buffer único `B`.

### Decisiones (input del simulador)
- **Asignación**: `caliber_by_outlet[m] = c` (estática) o política dinámica para v2.

---

## 3) Modelos de mezcla intra-finca (importante)

Como los calibres llegan mezclados “aleatoriamente”, el simulador debe generar secuencias consistentes con los totales \(K_{f,c}\).

### Opción A — Urna sin reemplazo (recomendada)
Representar la finca `f` como una “urna” con bolas de distintos colores (calibres), con masa total conocida. En cada micro-lote, se extrae una cantidad proporcional a lo que queda, con ruido.

Implementación práctica:
- Mantener `remaining_kg_by_caliber`.
- En cada paso:
  - definir `q_total` kg que ingresan (o por tiempo, `arrival_rate * dt`).
  - samplear `q_c` de una multinomial usando probabilidades `remaining/remaining_total`, y luego recortar para no exceder remanentes.
  - restar del remanente.

Ventaja: garantiza que al terminar la finca se cumpla exactamente \(K_{f,c}\).

### Opción B — Proceso i.i.d. por proporciones (rápido, pero menos realista)
En cada paso, samplear usando \(p_{f,c}\) constante. No garantiza cumplir el total exacto por finca, salvo corrección final.

---

## 4) Arquitectura de simulación (SimPy)

### 4.1 Entidades SimPy
- `Environment`: reloj de simulación.
- `Store` por calibre (colas): `simpy.Store` (si modelás micro-lotes discretos) o `Container` (si modelás kg continuos).
- `Process` de llegada: genera micro-lotes y los encola en el calibre correspondiente.
- `Process` por salida: consume del calibre asignado y acumula producción.

**Recomendación:** usar `simpy.Container` por calibre porque el estado natural es “kg en cola”.

### 4.2 Modelado de buffers (bloqueo)
Con buffers pequeños:
- si `Q_c` llega a `B_c`, la llegada de ese calibre se frena o bloquea upstream.
- con mezcla, el upstream (finca) puede bloquearse completo si no puede “depositar” parte del micro-lote (depende del modelado físico real).

Dos variantes:
1) **Bloqueo total**: si cualquier calibre excede buffer, se detiene la entrada total (representa un transportador único que se satura).
2) **Bloqueo parcial**: solo se bloquea el calibre saturado (si hay separación posterior).

Si tu experiencia dice “se te explota el sistema y todo va al cuello”, suele parecerse a **bloqueo total**.

---

## 5) Eventos y dinámica

### 5.1 Proceso de llegada (por finca)
Para cada finca `f`:
1. Inicializar remanentes `R_c = K[f,c]`.
2. Mientras `sum(R_c) > 0` y `t < H`:
   - definir `q_total` que intenta ingresar en este paso.
   - samplear `q_c` por urna sin reemplazo.
   - aplicar lógica de buffer:
     - si bloqueo total: si para algún `c`, `Q_c + q_c > B_c`, entonces esperar (yield timeout) y reintentar con menor `q_total` o con el mismo.
     - si parcial: encolar hasta tope y dejar “remanente” para el siguiente paso.
   - hacer `Q_c.put(q_c)` o `container.put(q_c)`.

Al terminar la finca, pasar a la siguiente.

### 5.2 Proceso de cada salida
Para una salida `m` asignada a calibre `c`:
- mientras `t < H`:
  - si `Q_c` tiene inventario:
    - consumir `min(v_m * dt_hours, Q_c)` kg
    - acumular producción
  - si `Q_c` está vacío:
    - registrar ociosidad
  - avanzar tiempo `dt`

En SimPy, en vez de un “loop por dt” podés modelar eventos de consumo continuo, pero el loop por dt es suficiente en v0 y muy legible.

---

## 6) Métricas y reportes

### Métricas mínimas
- `total_packed_kg`
- `packed_kg_by_caliber[c]`
- `utilization_by_outlet[m]` = tiempo consumiendo / tiempo disponible
- `idle_time_by_outlet[m]`
- `blocked_time_infeed` (si modelás bloqueo)
- `max_queue_kg_by_caliber[c]`
- `queue_time_series` (opcional) para graficar

### Repeticiones
Correr N seeds:
- promedios, mediana, P10/P90 de throughput y bloqueo
- intervalos de confianza (bootstrap opcional)

---

## 7) Especificación de interfaces (Python)

### Tipos de datos
```python
from dataclasses import dataclass
from typing import Dict, List, Optional

Caliber = int
OutletType = str  # "AUTO"|"BULK"|"MANUAL"

@dataclass(frozen=True)
class PackingLineConfig:
    shift_hours: float
    dt_seconds: int
    outlet_types: List[OutletType]         # len = M
    speed_kgph: Dict[OutletType, float]    # {"AUTO":..., "BULK":..., "MANUAL":...}
    buffer_kg_by_caliber: Optional[Dict[Caliber, float]] = None
    total_buffer_blocking: bool = True     # bloqueo total vs parcial

@dataclass(frozen=True)
class FarmLot:
    farm_id: str
    kg_by_caliber: Dict[Caliber, float]    # {80:..., 100:..., 120:...}

@dataclass(frozen=True)
class Assignment:
    caliber_by_outlet: List[Caliber]       # len=M
```

### Resultados
```python
@dataclass
class SimulationResult:
    seed: int
    total_packed_kg: float
    packed_kg_by_caliber: Dict[Caliber, float]
    utilization_by_outlet: List[float]
    idle_time_by_outlet_hours: List[float]
    blocked_time_infeed_hours: float
    max_queue_kg_by_caliber: Dict[Caliber, float]
```

---

## 8) Skeleton de código (SimPy)

> Nota: Esto es un esqueleto “implementable” que podés copiar a `src/lemon_packing/sim/simpy_engine.py`.

```python
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import simpy

def _dt_hours(dt_seconds: int) -> float:
    return dt_seconds / 3600.0

def sample_micro_lot_urn(
    remaining: Dict[int, float],
    q_total: float,
    rng: random.Random,
) -> Dict[int, float]:
    # Probabilidades proporcionales al remanente.
    total_rem = sum(remaining.values())
    if total_rem <= 0:
        return {c: 0.0 for c in remaining}

    # Multinomial aproximada por fracciones + ruido suave.
    # Para mantenerlo simple y estable numéricamente:
    probs = {c: remaining[c] / total_rem for c in remaining}
    q_by_c = {c: q_total * probs[c] for c in remaining}

    # Ajuste: no exceder remanente
    for c in list(q_by_c.keys()):
        q_by_c[c] = min(q_by_c[c], remaining[c])

    # Normalizar si recortamos fuerte
    used = sum(q_by_c.values())
    if used < q_total and total_rem - used > 1e-9:
        # Repartir el sobrante según remanente residual
        residual = {c: max(remaining[c] - q_by_c[c], 0.0) for c in remaining}
        res_total = sum(residual.values())
        if res_total > 0:
            extra = q_total - used
            for c in residual:
                add = extra * (residual[c] / res_total)
                q_by_c[c] += min(add, residual[c])

    # Garantía final
    for c in q_by_c:
        q_by_c[c] = max(q_by_c[c], 0.0)

    return q_by_c

@dataclass
class _State:
    packed_by_caliber: Dict[int, float]
    busy_time_by_outlet_hours: List[float]
    idle_time_by_outlet_hours: List[float]
    blocked_time_infeed_hours: float
    max_queue_by_caliber: Dict[int, float]

def infeed_process(
    env: simpy.Environment,
    farms: List[FarmLot],
    queues: Dict[int, simpy.Container],
    config: PackingLineConfig,
    state: _State,
    seed: int,
    arrival_rate_kgph: float,
):
    rng = random.Random(seed)
    dt_h = _dt_hours(config.dt_seconds)

    # Buffers
    buffer = config.buffer_kg_by_caliber

    for farm in farms:
        remaining = dict(farm.kg_by_caliber)

        while env.now < config.shift_hours * 3600 and sum(remaining.values()) > 1e-6:
            q_total = arrival_rate_kgph * dt_h
            q_by_c = sample_micro_lot_urn(remaining, q_total, rng)

            # Check blocking
            blocked = False
            if buffer is not None:
                if config.total_buffer_blocking:
                    for c, q in q_by_c.items():
                        cap = buffer.get(c, float("inf"))
                        if queues[c].level + q > cap + 1e-9:
                            blocked = True
                            break
                else:
                    # partial blocking handled below by clipping
                    blocked = False

            if blocked:
                state.blocked_time_infeed_hours += dt_h
                yield env.timeout(config.dt_seconds)
                continue

            # Put into queues (with partial clip if enabled)
            for c, q in q_by_c.items():
                if q <= 0:
                    continue
                if buffer is not None and not config.total_buffer_blocking:
                    cap = buffer.get(c, float("inf"))
                    q = min(q, max(cap - queues[c].level, 0.0))
                if q > 0:
                    yield queues[c].put(q)
                    remaining[c] -= q

                    # Track max queue
                    state.max_queue_by_caliber[c] = max(state.max_queue_by_caliber.get(c, 0.0), queues[c].level)

            yield env.timeout(config.dt_seconds)

def outlet_process(
    env: simpy.Environment,
    outlet_idx: int,
    caliber: int,
    queues: Dict[int, simpy.Container],
    config: PackingLineConfig,
    state: _State,
):
    dt_h = _dt_hours(config.dt_seconds)
    v = config.speed_kgph[config.outlet_types[outlet_idx]]  # kg/h
    while env.now < config.shift_hours * 3600:
        available = queues[caliber].level
        can_pack = v * dt_h
        if available > 1e-9:
            qty = min(can_pack, available)
            yield queues[caliber].get(qty)
            state.packed_by_caliber[caliber] = state.packed_by_caliber.get(caliber, 0.0) + qty
            state.busy_time_by_outlet_hours[outlet_idx] += dt_h
        else:
            state.idle_time_by_outlet_hours[outlet_idx] += dt_h

        # track max queue each tick (optional)
        state.max_queue_by_caliber[caliber] = max(state.max_queue_by_caliber.get(caliber, 0.0), queues[caliber].level)
        yield env.timeout(config.dt_seconds)

def run_simulation(
    config: PackingLineConfig,
    farms: List[FarmLot],
    assignment: Assignment,
    arrival_rate_kgph: float,
    seed: int,
) -> SimulationResult:
    calibers = sorted({c for f in farms for c in f.kg_by_caliber.keys()})
    env = simpy.Environment()

    queues = {c: simpy.Container(env, init=0.0, capacity=float("inf")) for c in calibers}

    state = _State(
        packed_by_caliber={},
        busy_time_by_outlet_hours=[0.0] * len(config.outlet_types),
        idle_time_by_outlet_hours=[0.0] * len(config.outlet_types),
        blocked_time_infeed_hours=0.0,
        max_queue_by_caliber={c: 0.0 for c in calibers},
    )

    env.process(infeed_process(env, farms, queues, config, state, seed, arrival_rate_kgph))

    for m, c in enumerate(assignment.caliber_by_outlet):
        env.process(outlet_process(env, m, c, queues, config, state))

    env.run(until=config.shift_hours * 3600)

    total = sum(state.packed_by_caliber.values())
    utilization = [
        bt / config.shift_hours if config.shift_hours > 0 else 0.0
        for bt in state.busy_time_by_outlet_hours
    ]

    return SimulationResult(
        seed=seed,
        total_packed_kg=total,
        packed_kg_by_caliber=dict(state.packed_by_caliber),
        utilization_by_outlet=utilization,
        idle_time_by_outlet_hours=list(state.idle_time_by_outlet_hours),
        blocked_time_infeed_hours=state.blocked_time_infeed_hours,
        max_queue_kg_by_caliber=dict(state.max_queue_by_caliber),
    )
```

---

## 9) Notas de calibración (cómo ajustar el simulador a la realidad)
- `arrival_rate_kgph`: si la entrada es “sin restricción” hasta bloquear, ponelo alto (ej. 10x capacidad) para que el cuello lo determine el empaque.
- `dt_seconds`: 10–60 segundos suele funcionar; muy grande suaviza demasiado y muy chico aumenta costo computacional.
- `total_buffer_blocking`: elegir según si el transportador/entrada se bloquea completo al saturarse un calibre.
- Validación: comparar throughput promedio y cuellos con mediciones reales (si existen).

---

## 10) Checklist de “Definition of Done” del simulador
- [ ] Reproduce escenarios simples (1 finca, 1 calibre) con throughput = suma de capacidades asignadas.
- [ ] Con mezcla, muestra bloqueo cuando un calibre tiene capacidad insuficiente.
- [ ] Métricas exportadas a DataFrame.
- [ ] Soporta N seeds.
- [ ] Tests smoke: no negativos, conserva totales por finca (si urn model).
