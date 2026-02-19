"""
Optimizador v1: asignación estática que matchea proporciones de fruta con capacidad.

La idea operativa: se fijan las proporciones de limones por calibre (del preconteo/fincas)
y se organizan las máquinas para que la capacidad asignada a cada calibre coincida
lo mejor posible con esas proporciones.

Usa formulación robusta: maximiza lambda tal que para TODA finca, la capacidad
por calibre alcanza su parte del mix.
"""

from dataclasses import dataclass
from typing import List, Optional

import pulp

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


def recommend_assignment(
    config: PackingLineConfig,
    farms: List[FarmLot],
    first_kg_limit: Optional[float] = None,
    use_heuristic_if_M_above: int = 12,
) -> OptimizationResult:
    """
    Recomienda asignación calibre-salida que matchea proporciones con capacidad.

    Args:
        config: Configuración de la línea.
        farms: Lista de fincas (preconteo).
        first_kg_limit: Si se define, usa proporciones de los primeros N kg
            (orden de fincas). Ej: 80000 = optimizar para el tramo inicial.
        use_heuristic_if_M_above: Si hay más de N salidas, usa heurística greedy
            en lugar de MILP (mucho más rápido).

    Returns:
        OptimizationResult con caliber_by_outlet recomendado y lambda_star (kg/h).
    """
    M = len(config.outlet_types)
    calibers = sorted({c for f in farms for c in f.kg_by_caliber.keys()})
    v_m = [config.speed_kgph[t] for t in config.outlet_types]

    # Proporciones p[f,c] según alcance (todas las fincas o primeros N kg)
    p = _farms_to_proportions(farms, first_kg_limit=first_kg_limit, calibers=calibers)
    proportions_used = {c: sum(pf.get(c, 0) for pf in p) / len(p) for c in calibers} if p else {}

    if M > use_heuristic_if_M_above:
        # Heurística O(M*C): instantánea
        assignment, lambda_star = _recommend_assignment_heuristic(
            M, calibers, v_m, proportions_used
        )
        status = "Heuristic"
    else:
        # MILP exacto (lento con muchas salidas)
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
        model.solve(pulp.PULP_CBC_CMD(msg=False))
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
