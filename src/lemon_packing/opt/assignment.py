"""
Optimizador v1: asignación estática para maximizar kg procesados.

Objetivo: maximizar la cantidad de kg procesados en el turno.

Cuando se provee arrival_rate_kgph y seed, maximiza explícitamente total_packed_kg
simulando candidatos y eligiendo el mejor. Sin esos parámetros, usa lambda como proxy.
"""

from dataclasses import dataclass
from typing import List, Optional

import pulp

from lemon_packing.sim.simpy_engine import run_simulation
from lemon_packing.types import Assignment, FarmLot, PackingLineConfig


@dataclass
class OptimizationResult:
    """Resultado del optimizador de asignación."""

    caliber_by_outlet: List[int]
    lambda_star: float  # kg/h sostenibles
    status: str
    capacity_by_caliber: dict  # kg/h asignados por calibre
    proportions_used: Optional[dict] = None  # {c: pct} si aplica (primeros N kg o promedio)


def _farms_to_proportions(
    farms: List[FarmLot],
    first_kg_limit: Optional[float] = None,
    calibers: Optional[List[int]] = None,
) -> List[dict]:
    """
    Convierte fincas en proporciones por calibre.

    Si first_kg_limit está definido, simula tomar los primeros N kg en orden de fincas
    y devuelve las proporciones de ese tramo (ej. "primeros 80.000 kg").
    """
    if calibers is None:
        calibers = sorted({c for f in farms for c in f.kg_by_caliber.keys()})

    if first_kg_limit is None:
        # Todas las fincas: proporciones por finca
        return [
            {
                c: (farm.kg_by_caliber.get(c, 0) / total if total > 0 else 0)
                for c in calibers
            }
            for farm in farms
            for total in [sum(farm.kg_by_caliber.values())]
        ]

    # Primeros N kg: acumular en orden de fincas
    kg_by_c = {c: 0.0 for c in calibers}
    remaining = first_kg_limit
    for farm in farms:
        farm_total = sum(farm.kg_by_caliber.values())
        if remaining <= 0 or farm_total <= 0:
            break
        take = min(remaining, farm_total)
        for c in calibers:
            pct = farm.kg_by_caliber.get(c, 0) / farm_total
            kg_by_c[c] += take * pct
        remaining -= take
    total_taken = sum(kg_by_c.values())
    if total_taken <= 0:
        return [{c: 1.0 / len(calibers) for c in calibers}]  # fallback
    return [
        {c: (kg_by_c[c] / total_taken) for c in calibers}
    ]


def _recommend_assignment_mincost_style(
    M: int,
    calibers: List[int],
    v_m: List[float],
    p: List[dict],
) -> tuple:
    """
    Optimizador rápido tipo min-cost flow: búsqueda binaria en lambda + greedy.

    Equivalente al MILP pero O(log L * M) en lugar de resolver programación entera.
    Usa p_max[c] = max_f p_f[c] para la restricción robusta (todas las fincas).
    """
    if not p:
        return _recommend_assignment_heuristic(M, calibers, v_m, {c: 1.0 / len(calibers) for c in calibers})

    p_max = {c: max(pf.get(c, 0) for pf in p) for c in calibers}
    total_cap = sum(v_m)
    total_p_max = sum(p_max.values())
    if total_p_max <= 0:
        return _recommend_assignment_heuristic(M, calibers, v_m, {c: 1.0 / len(calibers) for c in calibers})

    # Lambda máximo teórico: total_cap (si toda la capacidad va a un calibre con p_max>0)
    lambda_hi = total_cap / min(p_max[c] for c in calibers if p_max[c] > 1e-9) if any(p_max[c] > 1e-9 for c in calibers) else total_cap
    lambda_lo = 0.0

    def _feasible(lam: float) -> tuple:
        """Devuelve (es_factible, asignación) si lam es factible."""
        d = {c: lam * p_max[c] for c in calibers}
        current_V = {c: 0.0 for c in calibers}
        outlet_order = sorted(range(M), key=lambda m: v_m[m], reverse=True)
        assignment = [0] * M
        for m in outlet_order:
            best_c = max(calibers, key=lambda c: d[c] - current_V[c])
            assignment[m] = best_c
            current_V[best_c] += v_m[m]
        ok = all(current_V[c] >= d[c] - 1e-6 for c in calibers)
        return ok, assignment

    # Búsqueda binaria
    best_assignment = None
    for _ in range(60):  # ~60 iteraciones para precisión numérica
        lam = (lambda_lo + lambda_hi) / 2
        ok, assignment = _feasible(lam)
        if ok:
            lambda_lo = lam
            best_assignment = assignment
        else:
            lambda_hi = lam

    if best_assignment is None:
        _, best_assignment = _feasible(lambda_lo)

    lambda_star = lambda_lo if best_assignment else 0.0
    return best_assignment, lambda_star


def _recommend_assignment_heuristic(
    M: int,
    calibers: List[int],
    v_m: List[float],
    proportions_used: dict,
) -> tuple:
    """
    Heurística greedy O(M*C): asigna cada salida al calibre con mayor déficit
    de capacidad respecto al target proporcional. Muy rápida para muchas salidas.
    """
    total_cap = sum(v_m)
    target_V = {c: proportions_used.get(c, 0) * total_cap for c in calibers}
    current_V = {c: 0.0 for c in calibers}

    # Ordenar salidas por velocidad descendente (las más rápidas primero)
    outlet_order = sorted(range(M), key=lambda m: v_m[m], reverse=True)

    assignment = [0] * M
    for m in outlet_order:
        best_c = max(calibers, key=lambda c: target_V[c] - current_V[c])
        assignment[m] = best_c
        current_V[best_c] += v_m[m]

    # Lambda* = min_c (V_c / p_c) cuando p_c > 0
    lambda_star = float("inf")
    for c in calibers:
        p = proportions_used.get(c, 0)
        if p > 1e-9 and current_V[c] > 0:
            lambda_star = min(lambda_star, current_V[c] / p)

    return assignment, lambda_star if lambda_star < float("inf") else 0.0


def _recommend_assignment_max_kg(
    config: PackingLineConfig,
    farms: List[FarmLot],
    arrival_rate_kgph: float,
    seed: int,
    first_kg_limit: Optional[float] = None,
    use_mincost_style: bool = False,
    milp_time_limit: Optional[int] = None,
    local_search: bool = True,
    max_local_search_passes: int = 2,
) -> OptimizationResult:
    """
    Maximiza explícitamente total_packed_kg: simula candidatos y elige el mejor.
    """
    M = len(config.outlet_types)
    calibers = sorted({c for f in farms for c in f.kg_by_caliber.keys()})
    v_m = [config.speed_kgph[t] for t in config.outlet_types]
    p = _farms_to_proportions(farms, first_kg_limit=first_kg_limit, calibers=calibers)
    proportions_used = {c: sum(pf.get(c, 0) for pf in p) / len(p) for c in calibers} if p else {}

    # Candidatos: optimizador (MILP o mincost) y heurística
    if use_mincost_style:
        opt_assignment, lambda_opt = _recommend_assignment_mincost_style(M, calibers, v_m, p)
        opt_status = "MinCostStyle"
    else:
        model = pulp.LpProblem("packing_assignment_v1", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", (range(M), calibers), lowBound=0, upBound=1, cat="Binary")
        lam = pulp.LpVariable("lambda", lowBound=0, cat="Continuous")
        model += lam
        for m in range(M):
            model += pulp.lpSum(x[m][c] for c in calibers) == 1
        for f in range(len(p)):
            for c in calibers:
                V_c = pulp.lpSum(v_m[m] * x[m][c] for m in range(M))
                model += V_c >= lam * p[f][c]
        model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=milp_time_limit or 25))
        opt_assignment = [max(calibers, key=lambda c: pulp.value(x[m][c]) or 0) for m in range(M)]
        lambda_opt = pulp.value(lam) or 0.0
        opt_status = pulp.LpStatus[model.status]

    heur_assignment, lambda_heur = _recommend_assignment_heuristic(M, calibers, v_m, proportions_used)

    # Simular ambos y elegir el que procese más kg
    r_opt = run_simulation(config, farms, Assignment(opt_assignment), arrival_rate_kgph, seed)
    r_heur = run_simulation(config, farms, Assignment(heur_assignment), arrival_rate_kgph, seed)

    if r_opt.total_packed_kg >= r_heur.total_packed_kg:
        best = list(opt_assignment)
        best_kg = r_opt.total_packed_kg
        status = f"MaxKg({opt_status})"
    else:
        best = list(heur_assignment)
        best_kg = r_heur.total_packed_kg
        status = "MaxKg(Heuristic)"

    # Búsqueda local: cambiar un outlet a la vez, simular, aceptar si mejora
    if local_search:
        for _ in range(max_local_search_passes):
            improved = False
            improved = False
            for m in range(M):
                for c in calibers:
                    if c == best[m]:
                        continue
                    old_c = best[m]
                    best[m] = c
                    r = run_simulation(config, farms, Assignment(best), arrival_rate_kgph, seed)
                    if r.total_packed_kg > best_kg:
                        best_kg = r.total_packed_kg
                        improved = True
                        status = "MaxKg(LocalSearch)"
                    else:
                        best[m] = old_c
            if not improved:
                break

    capacity_by_caliber = {c: 0.0 for c in calibers}
    for m, c in enumerate(best):
        capacity_by_caliber[c] += v_m[m]
    lambda_star = min(
        capacity_by_caliber[c] / pct if pct > 1e-9 else float("inf")
        for c, pct in proportions_used.items()
    ) if proportions_used else 0.0
    if lambda_star == float("inf"):
        lambda_star = 0.0

    return OptimizationResult(
        caliber_by_outlet=best,
        lambda_star=lambda_star,
        status=status,
        capacity_by_caliber=capacity_by_caliber,
        proportions_used=proportions_used,
    )


def recommend_assignment(
    config: PackingLineConfig,
    farms: List[FarmLot],
    first_kg_limit: Optional[float] = None,
    use_heuristic_if_M_above: int = 12,
    force_milp: bool = False,
    milp_time_limit: Optional[int] = None,
    use_mincost_style: bool = False,
    arrival_rate_kgph: Optional[float] = None,
    seed: Optional[int] = None,
    maximize_kg: bool = True,
    local_search: bool = True,
) -> OptimizationResult:
    """
    Recomienda asignación calibre-salida para maximizar kg procesados.

    Si arrival_rate_kgph y seed están definidos, maximiza explícitamente total_packed_kg
    simulando candidatos y eligiendo el mejor (y búsqueda local opcional).
    Si no, maximiza lambda (proxy de throughput).

    Args:
        config: Configuración de la línea.
        farms: Lista de fincas (preconteo).
        first_kg_limit: Si se define, usa proporciones de los primeros N kg.
        use_heuristic_if_M_above: Si hay más de N salidas, usa heurística (sin maximize_kg).
        force_milp: Si True, usa MILP en lugar de mincost.
        milp_time_limit: Límite de tiempo para el MILP (default 25).
        use_mincost_style: Si True, usa optimizador rápido en lugar de MILP.
        arrival_rate_kgph: Tasa de llegada (kg/h) para simulación.
        seed: Semilla para simulación reproducible.
        maximize_kg: Si True y hay arrival_rate+seed, maximiza kg explícitamente.

    Returns:
        OptimizationResult con caliber_by_outlet recomendado.
    """
    if maximize_kg and arrival_rate_kgph is not None and seed is not None:
        return _recommend_assignment_max_kg(
            config, farms, arrival_rate_kgph, seed,
            first_kg_limit=first_kg_limit,
            use_mincost_style=use_mincost_style,
            milp_time_limit=milp_time_limit,
            local_search=local_search,
        )

    M = len(config.outlet_types)
    calibers = sorted({c for f in farms for c in f.kg_by_caliber.keys()})
    v_m = [config.speed_kgph[t] for t in config.outlet_types]
    p = _farms_to_proportions(farms, first_kg_limit=first_kg_limit, calibers=calibers)
    proportions_used = {c: sum(pf.get(c, 0) for pf in p) / len(p) for c in calibers} if p else {}

    use_heuristic = not force_milp and M > use_heuristic_if_M_above
    if use_mincost_style and (force_milp or not use_heuristic):
        assignment, lambda_star = _recommend_assignment_mincost_style(M, calibers, v_m, p)
        status = "MinCostStyle"
    elif use_heuristic:
        assignment, lambda_star = _recommend_assignment_heuristic(
            M, calibers, v_m, proportions_used
        )
        status = "Heuristic"
    else:
        model = pulp.LpProblem("packing_assignment_v1", pulp.LpMaximize)
        x = pulp.LpVariable.dicts(
            "x", (range(M), calibers), lowBound=0, upBound=1, cat="Binary"
        )
        lam = pulp.LpVariable("lambda", lowBound=0, cat="Continuous")
        model += lam
        for m in range(M):
            model += pulp.lpSum(x[m][c] for c in calibers) == 1
        for f in range(len(p)):
            for c in calibers:
                V_c = pulp.lpSum(v_m[m] * x[m][c] for m in range(M))
                model += V_c >= lam * p[f][c]
        time_limit = milp_time_limit if milp_time_limit is not None else 25
        model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
        assignment = [
            max(calibers, key=lambda c: pulp.value(x[m][c]) or 0) for m in range(M)
        ]
        lambda_star = pulp.value(lam) or 0.0
        status = pulp.LpStatus[model.status]

    capacity_by_caliber = {c: 0.0 for c in calibers}
    for m, c in enumerate(assignment):
        capacity_by_caliber[c] += v_m[m]

    return OptimizationResult(
        caliber_by_outlet=assignment,
        lambda_star=lambda_star,
        status=status,
        capacity_by_caliber=capacity_by_caliber,
        proportions_used=proportions_used,
    )
