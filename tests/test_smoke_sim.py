"""
Tests smoke del simulador.

- No negativos en métricas
- Escenario 1 finca, 1 calibre: throughput ≈ capacidad asignada
- Conserva totales por finca (modelo urna)
"""

from lemon_packing.sim.simpy_engine import run_simulation
from lemon_packing.sim.generators import single_caliber_scenario, simple_scenario
from lemon_packing.types import PackingLineConfig, Assignment


def test_single_caliber_throughput():
    """1 finca, 1 calibre: throughput debería ser ~ capacidad de la salida."""
    farms, assignment = single_caliber_scenario(caliber=100, total_kg=50000)
    config = PackingLineConfig(
        shift_hours=8.0,
        dt_seconds=60,
        outlet_types=["AUTO"],
        speed_kgph={"AUTO": 1200, "BULK": 800, "MANUAL": 400},
        buffer_kg_by_outlet=None,
        total_buffer_blocking=True,
    )

    r = run_simulation(config, farms, assignment, arrival_rate_kgph=50000, seed=42)

    # Capacidad: 1200 kg/h * 8h = 9600 kg
    expected = 1200 * 8
    assert r.total_packed_kg >= 0.95 * expected, f"Throughput {r.total_packed_kg} < 0.95 * {expected}"
    assert r.total_packed_kg <= 1.05 * expected, f"Throughput {r.total_packed_kg} > 1.05 * {expected}"


def test_no_negatives():
    """Todas las métricas deben ser >= 0."""
    farms, assignment = simple_scenario()
    config = PackingLineConfig(
        shift_hours=8.0,
        dt_seconds=30,
        outlet_types=["AUTO", "BULK", "MANUAL"],
        speed_kgph={"AUTO": 1200, "BULK": 800, "MANUAL": 400},
        buffer_kg_by_outlet=None,
        total_buffer_blocking=True,
    )

    r = run_simulation(config, farms, assignment, arrival_rate_kgph=50000, seed=123)

    assert r.total_packed_kg >= 0
    assert r.blocked_time_infeed_hours >= 0
    assert all(u >= 0 for u in r.utilization_by_outlet)
    assert all(i >= 0 for i in r.idle_time_by_outlet_hours)
    assert all(q >= 0 for q in r.max_queue_kg_by_caliber.values())
    assert all(kg >= 0 for kg in r.packed_kg_by_caliber.values())


def test_totals_conserved_per_farm():
    """Con modelo urna, la suma embalada por calibre no puede exceder el total de las fincas."""
    farms, assignment = simple_scenario(total_kg=10000)
    total_available = sum(sum(f.kg_by_caliber.values()) for f in farms)

    config = PackingLineConfig(
        shift_hours=8.0,
        dt_seconds=30,
        outlet_types=["AUTO", "BULK", "MANUAL"],
        speed_kgph={"AUTO": 1200, "BULK": 800, "MANUAL": 400},
        buffer_kg_by_outlet=None,
        total_buffer_blocking=True,
    )

    r = run_simulation(config, farms, assignment, arrival_rate_kgph=50000, seed=42)

    packed_total = sum(r.packed_kg_by_caliber.values())
    assert packed_total <= total_available + 1e-6, f"Empacado {packed_total} > disponible {total_available}"
