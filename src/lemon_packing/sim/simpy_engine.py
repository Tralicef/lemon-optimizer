"""
Motor de simulación de eventos discretos con SimPy.

Reproduce el comportamiento operativo del empaque bajo:
- mezcla estocástica de calibres dentro de cada finca (modelo urna),
- buffers pequeños con bloqueo,
- múltiples salidas con distintas velocidades.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import simpy

from lemon_packing.types import (
    Assignment,
    FarmLot,
    PackingLineConfig,
    SimulationResult,
    Snapshot,
)


def _dt_hours(dt_seconds: int) -> float:
    """Convierte dt en segundos a fracción de hora."""
    return dt_seconds / 3600.0


def sample_micro_lot_urn(
    remaining: Dict[int, float],
    q_total: float,
    rng: random.Random,
) -> Dict[int, float]:
    """
    Samplea un micro-lote usando el modelo urna sin reemplazo con multinomial.

    Los calibres vienen MEZCLADOS: en cada micro-lote se extrae una cantidad
    aleatoria por calibre siguiendo una multinomial con probabilidades
    proporcionales al remanente. Esto produce mezcla estocástica (a veces más
    de un calibre, a veces de otro) respetando las cantidades totales K[f,c].
    """
    total_rem = sum(remaining.values())
    if total_rem <= 0:
        return {c: 0.0 for c in remaining}

    calibers = sorted(remaining.keys())
    probs = np.array([remaining[c] / total_rem for c in calibers])

    # Multinomial: repartir q_total en "unidades" con probabilidades
    # proporcionales al remanente. Produce mezcla estocástica real (a veces
    # más de un calibre, a veces de otro) respetando K[f,c].
    q_eff = min(q_total, total_rem)
    n_units = max(1, min(int(q_eff), 2000))  # resolución ~1 kg, cap por performance
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    counts = np_rng.multinomial(n_units, probs)

    # Convertir conteos a kg (suma = q_eff) y limitar al remanente
    q_by_c = {}
    for i, c in enumerate(calibers):
        q_raw = counts[i] * (q_eff / n_units) if n_units > 0 else 0
        q_by_c[c] = min(max(0, q_raw), remaining[c])

    # Ajuste: si por redondeo/clipping sobra capacidad, repartir según residual
    used = sum(q_by_c.values())
    if used < q_eff - 1e-9 and total_rem - used > 1e-9:
        residual = {c: max(remaining[c] - q_by_c[c], 0.0) for c in calibers}
        res_total = sum(residual.values())
        if res_total > 0:
            extra = min(q_eff - used, res_total)
            for c in calibers:
                add = extra * (residual[c] / res_total)
                q_by_c[c] += min(add, residual[c])

    return q_by_c


@dataclass
class _State:
    """Estado interno acumulado durante la simulación."""

    packed_by_caliber: Dict[int, float]
    packed_by_outlet: List[float]  # kg embalados por cada salida
    busy_time_by_outlet_hours: List[float]
    idle_time_by_outlet_hours: List[float]
    blocked_time_infeed_hours: float
    max_queue_by_caliber: Dict[int, float]
    entered_by_caliber: Dict[int, float]  # kg totales que han entrado por calibre
    current_farm_id: List[str]  # finca actualmente en procesamiento [0]=id (mutable)
    current_farm_remaining: Dict[int, float]  # kg que quedan de la finca actual por calibre


def infeed_process(
    env: simpy.Environment,
    farms: List[FarmLot],
    queues: Dict[int, simpy.Container],
    config: PackingLineConfig,
    state: _State,
    seed: int,
    arrival_rate_kgph: float,
    buffer_kg_by_caliber: Dict[int, float] | None = None,
):
    """Proceso de llegada: genera micro-lotes desde las fincas hacia las colas."""
    rng = random.Random(seed)
    dt_h = _dt_hours(config.dt_seconds)
    buffer = buffer_kg_by_caliber

    for farm in farms:
        state.current_farm_id[0] = farm.farm_id
        remaining = dict(farm.kg_by_caliber)
        state.current_farm_remaining.clear()
        state.current_farm_remaining.update(remaining)

        while env.now < config.shift_hours * 3600 and sum(remaining.values()) > 1e-6:
            q_total = arrival_rate_kgph * dt_h
            q_by_c = sample_micro_lot_urn(remaining, q_total, rng)

            # Verificar bloqueo de buffer
            blocked = False
            if buffer is not None and config.total_buffer_blocking:
                for c, q in q_by_c.items():
                    if q <= 0:
                        continue
                    cap = buffer.get(c, float("inf"))
                    current_level = queues[c].level
                    if current_level + q > cap + 1e-9:
                        blocked = True
                        break

            if blocked:
                state.blocked_time_infeed_hours += dt_h
                state.current_farm_remaining.clear()
                state.current_farm_remaining.update(remaining)
                yield env.timeout(config.dt_seconds)
                continue

            # Poner en colas de forma atómica (todos los calibres juntos)
            # así no hay intercalado con outlets/snapshot que vería mezcla parcial
            puts = []
            to_put = {}
            for c, q in list(q_by_c.items()):
                if q <= 1e-9:
                    continue

                if buffer is not None and not config.total_buffer_blocking:
                    cap = buffer.get(c, float("inf"))
                    space = max(cap - queues[c].level, 0.0)
                    q = min(q, space)

                if q > 1e-9:
                    puts.append(queues[c].put(q))
                    to_put[c] = q

            if puts:
                yield env.all_of(puts)
                for c, q in to_put.items():
                    remaining[c] -= q
                    state.entered_by_caliber[c] = (
                        state.entered_by_caliber.get(c, 0.0) + q
                    )
                    state.max_queue_by_caliber[c] = max(
                        state.max_queue_by_caliber.get(c, 0.0), queues[c].level
                    )

            state.current_farm_remaining.clear()
            state.current_farm_remaining.update(remaining)
            yield env.timeout(config.dt_seconds)


def outlet_process(
    env: simpy.Environment,
    outlet_idx: int,
    caliber: int,
    queues: Dict[int, simpy.Container],
    config: PackingLineConfig,
    state: _State,
):
    """Proceso de cada salida: consume del calibre asignado y acumula producción."""
    dt_h = _dt_hours(config.dt_seconds)
    outlet_type = config.outlet_types[outlet_idx]
    v = config.speed_kgph[outlet_type]  # kg/h

    while env.now < config.shift_hours * 3600:
        available = queues[caliber].level
        can_pack = v * dt_h

        if available > 1e-9:
            qty = min(can_pack, available)
            yield queues[caliber].get(qty)
            state.packed_by_caliber[caliber] = (
                state.packed_by_caliber.get(caliber, 0.0) + qty
            )
            state.packed_by_outlet[outlet_idx] += qty
            state.busy_time_by_outlet_hours[outlet_idx] += dt_h
        else:
            state.idle_time_by_outlet_hours[outlet_idx] += dt_h

        state.max_queue_by_caliber[caliber] = max(
            state.max_queue_by_caliber.get(caliber, 0.0), queues[caliber].level
        )
        yield env.timeout(config.dt_seconds)


def snapshot_process(
    env: simpy.Environment,
    queues: Dict[int, simpy.Container],
    state: _State,
    assignment: Assignment,
    config: PackingLineConfig,
    interval_minutes: float,
    seed: int,
    buffer_kg_by_caliber: Dict[int, float] | None,
    snapshots_out: List[Snapshot] | None = None,
):
    """Proceso que registra un snapshot del estado cada interval_minutes."""
    interval_seconds = interval_minutes * 60
    period_hours = interval_minutes / 60.0
    prev_busy = [0.0] * len(config.outlet_types)
    prev_total_packed = 0.0

    while env.now < config.shift_hours * 3600:
        t_hours = env.now / 3600.0
        total_packed = sum(state.packed_by_caliber.values())
        total_entered = sum(state.entered_by_caliber.values())

        # Capacidad efectiva último periodo (kg/h)
        capacity_last_kgph = (total_packed - prev_total_packed) / period_hours if period_hours > 1e-9 else 0

        # Cuello de botella: caliber con buffer más lleno = outlets saturados;
        # caliber con outlets más ociosos = falta suministro
        buffer_occ_by_c = {}
        util_by_c = {c: [] for c in sorted(queues.keys())}
        for c in sorted(queues.keys()):
            cap = buffer_kg_by_caliber.get(c, float("inf")) if buffer_kg_by_caliber else float("inf")
            level = queues[c].level
            if cap and cap < 1e9 and cap > 0:
                buffer_occ_by_c[c] = (level / cap * 100)
            else:
                buffer_occ_by_c[c] = 0
            for m in range(len(config.outlet_types)):
                if assignment.caliber_by_outlet[m] == c:
                    u = (state.busy_time_by_outlet_hours[m] - prev_busy[m]) / period_hours if period_hours > 1e-9 else 0
                    util_by_c[c].append(u)
        util_avg_by_c = {c: sum(util_by_c[c]) / len(util_by_c[c]) if util_by_c[c] else 0 for c in util_by_c}
        max_occ_c = max(buffer_occ_by_c, key=buffer_occ_by_c.get) if buffer_occ_by_c else None
        min_util_c = min(util_avg_by_c, key=util_avg_by_c.get) if util_avg_by_c else None
        if max_occ_c is not None and buffer_occ_by_c.get(max_occ_c, 0) > 70:
            bottleneck_str = f"Calibre {max_occ_c}: buffers al {buffer_occ_by_c[max_occ_c]:.0f}% (outlets saturados)"
        elif min_util_c is not None and util_avg_by_c.get(min_util_c, 1) < 0.5:
            bottleneck_str = f"Calibre {min_util_c}: outlets al {util_avg_by_c[min_util_c]:.0%} (falta fruta en mix)"
        else:
            bottleneck_str = "Sin cuello obvio"

        # Guardar snapshot para visualización
        if snapshots_out is not None:
            queue_kg = {c: float(queues[c].level) for c in sorted(queues.keys())}
            packed_c = dict(state.packed_by_caliber)
            packed_o = list(state.packed_by_outlet)
            entered_c = dict(state.entered_by_caliber)
            buf_cap = dict(buffer_kg_by_caliber) if buffer_kg_by_caliber else None
            remaining_kg = dict(state.current_farm_remaining)
            busy_hours = list(state.busy_time_by_outlet_hours)
            snapshots_out.append(
                Snapshot(
                    t_hours=t_hours,
                    farm_id=state.current_farm_id[0],
                    queue_kg_by_caliber=queue_kg,
                    packed_kg_by_caliber=packed_c,
                    packed_kg_by_outlet=packed_o,
                    entered_kg_by_caliber=entered_c,
                    remaining_kg_by_caliber=remaining_kg,
                    busy_hours_by_outlet=busy_hours,
                    bottleneck=bottleneck_str,
                    capacity_last_period_kgph=capacity_last_kgph,
                    buffer_capacity_by_caliber=buf_cap,
                    blocked_hours=state.blocked_time_infeed_hours,
                )
            )

        # % de lo que ha entrado por calibre
        pct_entered = []
        for c in sorted(state.entered_by_caliber.keys()):
            kg = state.entered_by_caliber.get(c, 0)
            pct = (kg / total_entered * 100) if total_entered > 1e-9 else 0
            pct_entered.append(f"c{c}={pct:.1f}%")
        pct_entered_str = ", ".join(pct_entered) if pct_entered else "-"

        # Ocupación de buffers (%)
        buffer_occ = []
        if buffer_kg_by_caliber:
            for c in sorted(queues.keys()):
                cap = buffer_kg_by_caliber.get(c, float("inf"))
                level = queues[c].level
                if cap > 0 and cap < float("inf"):
                    occ = (level / cap * 100) if cap > 1e-9 else 0
                    buffer_occ.append(f"c{c}={occ:.0f}%")
                else:
                    buffer_occ.append(f"c{c}={level:.0f}kg")
        buffer_occ_str = ", ".join(buffer_occ) if buffer_occ else "sin límite"

        # Embalado por cada máquina
        packed_by_machine = ", ".join(
            f"#{m}(c{assignment.caliber_by_outlet[m]}):{state.packed_by_outlet[m]:.0f}kg"
            for m in range(len(config.outlet_types))
        )

        # Lo que queda de la finca (kg y %)
        rem_total = sum(state.current_farm_remaining.values())
        rem_str = ", ".join(
            f"c{c}={state.current_farm_remaining.get(c, 0):.0f}kg"
            for c in sorted(state.current_farm_remaining.keys())
        )
        rem_pct_str = ", ".join(
            f"c{c}={100*state.current_farm_remaining.get(c, 0)/rem_total:.1f}%"
            for c in sorted(state.current_farm_remaining.keys())
        ) if rem_total > 1e-9 else "-"

        lines = [
            "",
            f"--- Snapshot @ {t_hours:.1f}h (seed {seed}) ---",
            f"  Finca en proceso: {state.current_farm_id[0]}",
            f"  Queda de finca (kg): {rem_str}",
            f"  Queda de finca (%): {rem_pct_str}",
            f"  % entrada por calibre: {pct_entered_str}",
            f"  Cola (kg): " + ", ".join(
                f"c{c}={queues[c].level:.0f}" for c in sorted(queues.keys())
            ),
            f"  Ocupación buffers: {buffer_occ_str}",
            f"  Embalado total: {total_packed:,.0f} kg",
            f"  Por calibre: " + ", ".join(
                f"c{c}={state.packed_by_caliber.get(c, 0):.0f}"
                for c in sorted(state.packed_by_caliber.keys())
            ),
            f"  Por máquina: {packed_by_machine}",
            f"  Tiempo bloqueo infeed: {state.blocked_time_infeed_hours:.2f} h",
            f"  Cuello de botella: {bottleneck_str}",
            f"  Capacidad último periodo: {capacity_last_kgph:,.0f} kg/h",
            "  Utilización último periodo: " + ", ".join(
                f"#{m}:{(state.busy_time_by_outlet_hours[m]-prev_busy[m])/period_hours:.0%}"
                for m in range(len(config.outlet_types))
            ),
            "  Utilización acumulada: " + ", ".join(
                f"#{m}:{state.busy_time_by_outlet_hours[m]/max(t_hours,0.01):.0%}"
                for m in range(len(config.outlet_types))
            ),
        ]
        logging.info("\n".join(lines))

        prev_busy = list(state.busy_time_by_outlet_hours)
        prev_total_packed = total_packed
        yield env.timeout(interval_seconds)


def run_simulation(
    config: PackingLineConfig,
    farms: List[FarmLot],
    assignment: Assignment,
    arrival_rate_kgph: float,
    seed: int,
    snapshot_interval_minutes: float | None = None,
    snapshots_out: List[Snapshot] | None = None,
) -> SimulationResult:
    """
    Ejecuta una simulación completa del turno de empaque.

    Args:
        config: Configuración de la línea.
        farms: Lista de lotes por finca.
        assignment: Asignación calibre por salida.
        arrival_rate_kgph: Tasa de llegada en kg/h.
        seed: Semilla para reproducibilidad.

    Returns:
        SimulationResult con todas las métricas.
    """
    calibers = sorted({c for f in farms for c in f.kg_by_caliber.keys()})
    env = simpy.Environment()

    queues = {
        c: simpy.Container(env, init=0.0, capacity=float("inf")) for c in calibers
    }

    # Derivar buffer por calibre desde buffer por salida
    buffer_kg_by_caliber: Dict[int, float] | None = None
    if config.buffer_kg_by_outlet is not None:
        buffer_kg_by_caliber = {c: 0.0 for c in calibers}
        for m, c in enumerate(assignment.caliber_by_outlet):
            if c in buffer_kg_by_caliber and m < len(config.buffer_kg_by_outlet):
                buffer_kg_by_caliber[c] += config.buffer_kg_by_outlet[m]

    state = _State(
        packed_by_caliber={c: 0.0 for c in calibers},
        packed_by_outlet=[0.0] * len(config.outlet_types),
        busy_time_by_outlet_hours=[0.0] * len(config.outlet_types),
        idle_time_by_outlet_hours=[0.0] * len(config.outlet_types),
        blocked_time_infeed_hours=0.0,
        max_queue_by_caliber={c: 0.0 for c in calibers},
        entered_by_caliber={c: 0.0 for c in calibers},
        current_farm_id=["(inicio)"],
        current_farm_remaining={c: 0.0 for c in calibers},
    )

    env.process(
        infeed_process(
            env, farms, queues, config, state, seed, arrival_rate_kgph,
            buffer_kg_by_caliber=buffer_kg_by_caliber,
        )
    )

    for m, c in enumerate(assignment.caliber_by_outlet):
        env.process(outlet_process(env, m, c, queues, config, state))

    run_snapshots = (snapshot_interval_minutes and snapshot_interval_minutes > 0) or snapshots_out is not None
    if run_snapshots:
        interval = snapshot_interval_minutes if snapshot_interval_minutes else 10.0
        env.process(
            snapshot_process(
                env, queues, state, assignment, config,
                interval, seed,
                buffer_kg_by_caliber,
                snapshots_out=snapshots_out,
            )
        )

    env.run(until=config.shift_hours * 3600)

    # Snapshot final para visualización
    if snapshots_out is not None:
        queue_kg = {c: float(queues[c].level) for c in sorted(queues.keys())}
        buf_cap = dict(buffer_kg_by_caliber) if buffer_kg_by_caliber else None
        snapshots_out.append(
            Snapshot(
                t_hours=config.shift_hours,
                farm_id=state.current_farm_id[0],
                queue_kg_by_caliber=queue_kg,
                packed_kg_by_caliber=dict(state.packed_by_caliber),
                packed_kg_by_outlet=list(state.packed_by_outlet),
                entered_kg_by_caliber=dict(state.entered_by_caliber),
                remaining_kg_by_caliber=dict(state.current_farm_remaining),
                busy_hours_by_outlet=list(state.busy_time_by_outlet_hours),
                buffer_capacity_by_caliber=buf_cap,
                blocked_hours=state.blocked_time_infeed_hours,
            )
        )

    total = sum(state.packed_by_caliber.values())
    utilization = [
        bt / config.shift_hours if config.shift_hours > 0 else 0.0
        for bt in state.busy_time_by_outlet_hours
    ]

    return SimulationResult(
        seed=seed,
        total_packed_kg=total,
        packed_kg_by_caliber=dict(state.packed_by_caliber),
        packed_kg_by_outlet=list(state.packed_by_outlet),
        utilization_by_outlet=utilization,
        idle_time_by_outlet_hours=list(state.idle_time_by_outlet_hours),
        blocked_time_infeed_hours=state.blocked_time_infeed_hours,
        max_queue_kg_by_caliber=dict(state.max_queue_by_caliber),
    )
