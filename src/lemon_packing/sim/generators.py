"""Generadores de datos de prueba para el simulador."""

from lemon_packing.types import Assignment, FarmLot


def simple_scenario(
    total_kg: float = 16000,
    calibers: list[int] = [80, 100, 120],
) -> tuple[list[FarmLot], Assignment]:
    """
    Genera un escenario simple: 1 finca, mezcla uniforme, asignación balanceada.

    Útil para validación smoke: 1 finca, 1 calibre debería dar throughput
    igual a suma de capacidades asignadas.
    """
    kg_per_c = total_kg / len(calibers)
    kg_by_caliber = {c: kg_per_c for c in calibers}
    farms = [FarmLot(farm_id="F1", kg_by_caliber=kg_by_caliber)]

    # 3 salidas, una por calibre (simplificado)
    caliber_by_outlet = calibers
    assignment = Assignment(caliber_by_outlet=caliber_by_outlet)

    return farms, assignment


def single_caliber_scenario(
    caliber: int = 100,
    total_kg: float = 10000,
) -> tuple[list[FarmLot], Assignment]:
    """
    Escenario con un solo calibre. Útil para validar que throughput = capacidad.
    """
    farms = [FarmLot(farm_id="F1", kg_by_caliber={caliber: total_kg})]
    assignment = Assignment(caliber_by_outlet=[caliber])
    return farms, assignment
