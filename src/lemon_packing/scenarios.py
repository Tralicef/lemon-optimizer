"""
Escenarios predefinidos de fincas con distintos niveles de variabilidad (Jensen-Shannon).

Basados en datos reales de referencia (24 fincas, mix ~24% c120, ~61% c100, ~15% c80):
- Muy bajo: datos suavizados hacia la media (fincas muy similares)
- Bajo: variación reducida
- Medio: datos originales
- Alto: variación amplificada (fincas más distintas)
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from lemon_packing.types import FarmLot


def jensen_shannon_index(farms: List[FarmLot], calibers: List[int] = [80, 100, 120]) -> float:
    """
    Índice de Jensen-Shannon para un conjunto de fincas.

    Calcula el promedio de la divergencia JSD entre todos los pares de fincas.
    Mide qué tan diferentes son las composiciones (proporciones por calibre) entre fincas.
    Valor en [0, 1]: 0 = todas iguales, 1 = máximamente diferentes.

    Returns:
        Promedio de JSD entre pares de fincas (distancia, sqrt de divergencia).
    """
    if not farms or len(farms) < 2:
        return 0.0

    # Convertir cada finca a vector de proporciones
    props = []
    for f in farms:
        total = sum(f.kg_by_caliber.get(c, 0) for c in calibers)
        if total <= 0:
            continue
        p = np.array([f.kg_by_caliber.get(c, 0) / total for c in calibers], dtype=float)
        props.append(p)

    if len(props) < 2:
        return 0.0

    def _jsd(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon distance entre dos distribuciones."""
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        p = p / p.sum()
        q = q / q.sum()
        m = (p + q) / 2
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        divergence = 0.5 * (kl_pm + kl_qm)
        return np.sqrt(divergence)  # distancia JSD en [0, 1]

    total_jsd = 0.0
    count = 0
    for i in range(len(props)):
        for j in range(i + 1, len(props)):
            total_jsd += _jsd(props[i], props[j])
            count += 1

    return total_jsd / count if count > 0 else 0.0


@dataclass
class Scenario:
    """Escenario predefinido con fincas y su índice JSD."""

    id: str
    name: str
    description: str
    farms: List[FarmLot]
    jensen_shannon: float


TARGET_KG_PER_SCENARIO = 180_000


def _scale_farms_to_total(farms: List[FarmLot], target_kg: float = TARGET_KG_PER_SCENARIO) -> List[FarmLot]:
    """Escala las fincas para que el total sea target_kg, preservando proporciones."""
    total = sum(sum(f.kg_by_caliber.values()) for f in farms)
    if total <= 0:
        return farms
    scale = target_kg / total
    scaled = []
    for f in farms:
        new_kg = {c: v * scale for c, v in f.kg_by_caliber.items()}
        scaled.append(FarmLot(farm_id=f.farm_id, kg_by_caliber=new_kg))
    return scaled


def _make_scenarios() -> List[Scenario]:
    """Crea los 4 escenarios basados en datos reales. 24 fincas, 180.000 kg total."""
    calibers = [80, 100, 120]

    # Base: datos reales de referencia (120, 100, 80) - mix ~24% 120, ~61% 100, ~15% 80
    _base_raw = [
        (19606.35, 60943.93, 6007.15),
        (3577.24, 9548.93, 1603.69),
        (3599.06, 17745.67, 5540.48),
        (1370.76, 5845.50, 1525.88),
        (618.48, 2806.33, 671.65),
        (1436.47, 8512.84, 4146.09),
        (4047.85, 12557.38, 1323.28),
        (7156.0, 14857.0, 3682.0),
        (2499.08, 11738.12, 216.26),
        (3431.43, 9885.05, 141.21),
        (3911.89, 13112.10, 81.91),
        (3806.37, 13678.44, 93.81),
        (4082.92, 14051.21, 88.99),
        (6685.83, 30197.47, 570.99),
        (2140.07, 6077.64, 4309.30),
        (2783.07, 5754.08, 4627.33),
        (3467.45, 4766.65, 3793.22),
        (4069.0, 5800.15, 3228.96),
        (6129.04, 5173.71, 1420.35),
        (3406.26, 2899.14, 834.21),
        (2145.51, 3521.91, 2521.52),
        (3600.08, 6654.82, 4917.62),
        (3109.06, 4741.97, 4331.37),
        (19454.38, 29134.06, 17028.55),
    ]

    def _to_farms(rows, col_order=(120, 100, 80)):
        """Convierte filas (c120, c100, c80) en FarmLot."""
        return [FarmLot(f"F{i+1}", {col_order[j]: rows[i][j] for j in range(3)}) for i in range(len(rows))]

    def _smooth_to_proportion(rows, target_prop, strength=0.0):
        """Interpola cada finca hacia target_prop. strength=0 = sin cambio, 1 = todas iguales."""
        out = []
        for i, r in enumerate(rows):
            total = sum(r)
            if total <= 0:
                continue
            p_act = (r[0] / total, r[1] / total, r[2] / total)
            p_new = (
                p_act[0] * (1 - strength) + target_prop[0] * strength,
                p_act[1] * (1 - strength) + target_prop[1] * strength,
                p_act[2] * (1 - strength) + target_prop[2] * strength,
            )
            s = sum(p_new)
            p_new = (p_new[0] / s, p_new[1] / s, p_new[2] / s)
            out.append((p_new[0] * total, p_new[1] * total, p_new[2] * total))
        return out

    def _amplify_variation(rows, factor=1.5):
        """Amplifica la variación: aleja proporciones del centro."""
        total_kg = sum(sum(r) for r in rows)
        avg_p = (
            sum(r[0] for r in rows) / total_kg,
            sum(r[1] for r in rows) / total_kg,
            sum(r[2] for r in rows) / total_kg,
        )
        out = []
        for r in rows:
            t = sum(r)
            if t <= 0:
                continue
            p = (r[0] / t, r[1] / t, r[2] / t)
            p_new = (
                avg_p[0] + (p[0] - avg_p[0]) * factor,
                avg_p[1] + (p[1] - avg_p[1]) * factor,
                avg_p[2] + (p[2] - avg_p[2]) * factor,
            )
            p_new = tuple(max(0.01, min(0.98, x)) for x in p_new)
            s = sum(p_new)
            p_new = (p_new[0] / s, p_new[1] / s, p_new[2] / s)
            out.append((p_new[0] * t, p_new[1] * t, p_new[2] * t))
        return out

    # Proporción media del base: ~24% 120, ~61% 100, ~15% 80
    _tot = sum(sum(r) for r in _base_raw)
    _avg_p = (sum(r[0] for r in _base_raw) / _tot, sum(r[1] for r in _base_raw) / _tot, sum(r[2] for r in _base_raw) / _tot)

    # Muy bajo: suavizar hacia la media (fincas más parecidas)
    esc_muy_bajo = _scale_farms_to_total(_to_farms(_smooth_to_proportion(_base_raw, _avg_p, strength=0.85)))

    # Bajo: suavizar un poco
    esc_bajo = _scale_farms_to_total(_to_farms(_smooth_to_proportion(_base_raw, _avg_p, strength=0.4)))

    # Medio: datos base exactos
    esc_medio = _scale_farms_to_total(_to_farms(_base_raw))

    # Alto: amplificar variación
    esc_alto = _scale_farms_to_total(_to_farms(_amplify_variation(_base_raw, factor=1.8)))

    scenarios = [
        Scenario(
            id="muy_bajo",
            name="Muy bajo",
            description="24 fincas (datos reales suavizados). Mix ~24% c120, ~61% c100, ~15% c80. 180.000 kg.",
            farms=esc_muy_bajo,
            jensen_shannon=jensen_shannon_index(esc_muy_bajo, calibers),
        ),
        Scenario(
            id="bajo",
            name="Bajo",
            description="24 fincas (datos reales, variación reducida). 180.000 kg total.",
            farms=esc_bajo,
            jensen_shannon=jensen_shannon_index(esc_bajo, calibers),
        ),
        Scenario(
            id="medio",
            name="Medio",
            description="24 fincas (datos reales de referencia). 180.000 kg total.",
            farms=esc_medio,
            jensen_shannon=jensen_shannon_index(esc_medio, calibers),
        ),
        Scenario(
            id="alto",
            name="Alto",
            description="24 fincas (datos reales con variación amplificada). 180.000 kg total.",
            farms=esc_alto,
            jensen_shannon=jensen_shannon_index(esc_alto, calibers),
        ),
    ]

    return scenarios


PREDEFINED_SCENARIOS = _make_scenarios()

# Etiqueta para el análisis que promedia los 4 escenarios con el mismo peso
ANALISIS_COMBINADO_LABEL = "Combinado (4 escenarios, peso igual)"


def merge_scenarios_equal_weight(
    scenarios: List[Scenario],
    calibers: List[int] = None,
    target_kg: float = TARGET_KG_PER_SCENARIO,
) -> List[FarmLot]:
    """
    Promedia los kg por finca y calibre entre todos los escenarios (mismo peso cada uno)
    y reescala el total a target_kg. Misma estructura de fincas (mismos farm_id) en cada escenario.
    """
    if calibers is None:
        calibers = [80, 100, 120]
    if not scenarios:
        return []
    n = len(scenarios)
    ids = [f.farm_id for f in scenarios[0].farms]
    merged: List[FarmLot] = []
    for fid in ids:
        kg_acc = {c: 0.0 for c in calibers}
        for scen in scenarios:
            farm = next((f for f in scen.farms if f.farm_id == fid), None)
            if farm is None:
                continue
            for c in calibers:
                kg_acc[c] += farm.kg_by_caliber.get(c, 0.0)
        for c in calibers:
            kg_acc[c] /= float(n)
        merged.append(FarmLot(farm_id=fid, kg_by_caliber=kg_acc))
    return _scale_farms_to_total(merged, target_kg)
