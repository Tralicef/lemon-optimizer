#!/usr/bin/env python3
"""
Punto de entrada del simulador de empaque de limones.

Ejecuta la simulación usando los archivos de configuración en configs/.
Las fincas se cargan desde un CSV (data/farms.csv por defecto).
"""

import logging
import sys
from pathlib import Path

from lemon_packing.io.loaders import load_farms_from_csv, load_simulator_config
from lemon_packing.types import Assignment
from lemon_packing.sim.metrics import print_summary, results_to_dataframe
from lemon_packing.sim.simpy_engine import run_simulation
from lemon_packing.opt import recommend_assignment


def main() -> None:
    """Ejecuta el simulador con la configuración por defecto."""
    base = Path(__file__).parent
    config_path = base / "configs" / "simulator_config.yaml"
    default_csv = base / "data" / "farms.csv"
    farms_csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv

    # Cargar configuración (incluye asignación de calibres)
    config, extra = load_simulator_config(config_path)
    farms = load_farms_from_csv(farms_csv_path)
    assignment = Assignment(caliber_by_outlet=extra["caliber_by_outlet"])

    arrival_rate = extra["arrival_rate_kgph"]
    n_seeds = extra["n_seeds"]
    seed_base = extra["seed_base"]

    # Validar que asignación coincide con número de salidas
    if len(assignment.caliber_by_outlet) != len(config.outlet_types):
        raise ValueError(
            f"Asignación tiene {len(assignment.caliber_by_outlet)} salidas, "
            f"pero config tiene {len(config.outlet_types)}"
        )

    # Optimizador: primeros 80 000 kg únicamente
    opt_result_first = recommend_assignment(config, farms, first_kg_limit=80000)

    print("\n" + "=" * 60)
    print("CONFIGURACIÓN INICIAL")
    print("=" * 60)
    print(f"Turno: {config.shift_hours}h | Seed única: {seed_base}")

    # Proporciones esperadas (promedio ponderado por finca)
    total_kg = sum(sum(f.kg_by_caliber.values()) for f in farms)
    prop_esperadas = {}
    for c in sorted({c for f in farms for c in f.kg_by_caliber.keys()}):
        kg_c = sum(f.kg_by_caliber.get(c, 0) for f in farms)
        prop_esperadas[c] = (kg_c / total_kg * 100) if total_kg > 0 else 0
    print(f"\nProporciones esperadas de fruta: " +
          ", ".join(f"c{c}={prop_esperadas[c]:.1f}%" for c in sorted(prop_esperadas)))

    print(f"\n--- Tu asignación (usada en simulación) ---")
    for m, c in enumerate(assignment.caliber_by_outlet):
        tipo = config.outlet_types[m]
        vel = config.speed_kgph[tipo]
        print(f"  Salida #{m}: calibre {c} ({tipo} @ {vel} kg/h)")
    cap_dueño = {c: 0 for c in prop_esperadas}
    for m, c in enumerate(assignment.caliber_by_outlet):
        cap_dueño[c] += config.speed_kgph[config.outlet_types[m]]
    total_cap = sum(cap_dueño.values())
    if total_cap > 0:
        print(f"  Capacidad por calibre: " +
              ", ".join(f"c{c}={cap_dueño[c]:.0f} kg/h ({100*cap_dueño[c]/total_cap:.1f}%)"
                        for c in sorted(cap_dueño)))

    print(f"\n--- Recomendación del optimizador (basada en primeros 80 000 kg) ---")
    prop_first = opt_result_first.proportions_used or {}
    print(f"  Proporciones de ese tramo: " +
          ", ".join(f"c{c}={100*prop_first.get(c,0):.1f}%" for c in sorted(prop_first)))
    for m, c in enumerate(opt_result_first.caliber_by_outlet):
        tipo = config.outlet_types[m]
        vel = config.speed_kgph[tipo]
        print(f"  Salida #{m}: calibre {c} ({tipo} @ {vel} kg/h)")
    total_cap_first = sum(opt_result_first.capacity_by_caliber.values())
    if total_cap_first > 0:
        print(f"  Capacidad por calibre: " +
              ", ".join(f"c{c}={opt_result_first.capacity_by_caliber[c]:.0f} kg/h "
                       f"({100*opt_result_first.capacity_by_caliber[c]/total_cap_first:.1f}%)"
                        for c in sorted(opt_result_first.capacity_by_caliber)))
    pct_max_first = (100 * opt_result_first.lambda_star / total_cap_first) if total_cap_first > 0 else 0
    print(f"  λ* = {opt_result_first.lambda_star:,.0f} kg/h sostenibles ({pct_max_first:.1f}% del máximo) | Estado: {opt_result_first.status}")

    print("\nModelo de llegada: los calibres vienen MEZCLADOS (distribución")
    print("multinomial proporcional al remanente en cada micro-lote).")
    print(f"\nFincas a procesar (en orden): {[f.farm_id for f in farms]}")
    for f in farms:
        total = sum(f.kg_by_caliber.values())
        pcts = {c: (kg / total * 100) if total > 0 else 0 for c, kg in f.kg_by_caliber.items()}
        print(f"  • {f.farm_id}: {total:,.0f} kg total | % por calibre: " +
              ", ".join(f"c{c}={pcts[c]:.1f}%" for c in sorted(pcts)))
    print("=" * 60 + "\n")
    sys.stdout.flush()

    snapshot_interval = extra.get("snapshot_interval_minutes")
    if snapshot_interval:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )

    # Ejecutar N simulaciones
    results = []
    for i in range(n_seeds):
        seed = seed_base + i
        r = run_simulation(
            config, farms, assignment, arrival_rate, seed,
            snapshot_interval_minutes=snapshot_interval,
        )
        results.append(r)

    # Reportar
    print_summary(results, str(config_path))

    # Exportar a DataFrame (útil para análisis)
    df = results_to_dataframe(results)
    out_path = base / "simulation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Resultados guardados en {out_path}")


if __name__ == "__main__":
    main()
