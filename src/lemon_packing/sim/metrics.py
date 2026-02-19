"""Métricas y reportes de simulación."""

from typing import List

import pandas as pd

from lemon_packing.types import SimulationResult


def results_to_dataframe(results: List[SimulationResult]) -> pd.DataFrame:
    """
    Convierte una lista de SimulationResult a un DataFrame con métricas agregadas.

    Incluye promedios, mediana, P10, P90 por métrica cuando hay múltiples seeds.
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        row = {
            "seed": r.seed,
            "total_packed_kg": r.total_packed_kg,
            "blocked_time_infeed_hours": r.blocked_time_infeed_hours,
        }
        for i, util in enumerate(r.utilization_by_outlet):
            row[f"util_outlet_{i}"] = util
        for i, idle in enumerate(r.idle_time_by_outlet_hours):
            row[f"idle_hours_outlet_{i}"] = idle
        for c, kg in r.packed_kg_by_caliber.items():
            row[f"packed_kg_c{c}"] = kg
        for m, kg in enumerate(r.packed_kg_by_outlet):
            row[f"packed_kg_outlet_{m}"] = kg
        for c, q in r.max_queue_kg_by_caliber.items():
            row[f"max_queue_kg_c{c}"] = q
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(results: List[SimulationResult], config_path: str = "") -> None:
    """Imprime un resumen legible de los resultados de simulación."""
    if not results:
        print("Sin resultados.")
        return

    r0 = results[0]
    print("\n" + "=" * 60)
    print("RESUMEN DE SIMULACIÓN")
    if config_path:
        print(f"Config: {config_path}")
    print("=" * 60)

    if len(results) == 1:
        r = r0
        print(f"\nSeed: {r.seed}")
        print(f"Throughput total: {r.total_packed_kg:,.0f} kg")
        print(f"\nEmbalado por calibre:")
        for c, kg in sorted(r.packed_kg_by_caliber.items()):
            print(f"  Calibre {c}: {kg:,.0f} kg")
        print(f"\nEmbalado por máquina (salida):")
        for m, kg in enumerate(r.packed_kg_by_outlet):
            print(f"  Salida #{m}: {kg:,.0f} kg")
        print(f"\nUtilización por salida: {[f'{u:.1%}' for u in r.utilization_by_outlet]}")
        print(f"Tiempo ocioso por salida (h): {[f'{x:.2f}' for x in r.idle_time_by_outlet_hours]}")
        print(f"Tiempo bloqueo infeed: {r.blocked_time_infeed_hours:.2f} h")
        print(f"\nCola máxima por calibre (kg): {r.max_queue_kg_by_caliber}")
    else:
        totals = [r.total_packed_kg for r in results]
        blocked = [r.blocked_time_infeed_hours for r in results]
        print(f"\nSeeds: {len(results)}")
        print(f"Throughput total: mean={sum(totals)/len(totals):,.0f} kg, "
              f"min={min(totals):,.0f}, max={max(totals):,.0f}")
        print(f"Tiempo bloqueo infeed: mean={sum(blocked)/len(blocked):.2f} h")

    print("=" * 60 + "\n")
