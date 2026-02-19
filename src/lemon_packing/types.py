"""Tipos de datos para el simulador y optimizador."""

from dataclasses import dataclass
from typing import Dict, List, Optional

Caliber = int
OutletType = str  # "AUTO" | "BULK" | "MANUAL"


@dataclass(frozen=True)
class PackingLineConfig:
    """Configuración de la línea de empaque."""

    shift_hours: float
    dt_seconds: int
    outlet_types: List[OutletType]  # len = M
    speed_kgph: Dict[OutletType, float]  # {"AUTO":..., "BULK":..., "MANUAL":...}
    buffer_kg_by_outlet: Optional[List[float]] = None  # kg máximo por salida (índice = outlet)
    total_buffer_blocking: bool = True  # bloqueo total vs parcial


@dataclass(frozen=True)
class FarmLot:
    """Lote de una finca con kg por calibre."""

    farm_id: str
    kg_by_caliber: Dict[Caliber, float]  # {80:..., 100:..., 120:...}


@dataclass(frozen=True)
class Assignment:
    """Asignación de calibre por salida."""

    caliber_by_outlet: List[Caliber]  # len=M


@dataclass
class Snapshot:
    """Estado del sistema en un instante (para visualización)."""

    t_hours: float
    farm_id: str
    queue_kg_by_caliber: Dict[Caliber, float]  # kg en cola por calibre
    packed_kg_by_caliber: Dict[Caliber, float]  # embalado acumulado por calibre
    packed_kg_by_outlet: List[float]  # embalado acumulado por máquina
    entered_kg_by_caliber: Dict[Caliber, float]  # kg que han entrado por calibre
    remaining_kg_by_caliber: Dict[Caliber, float]  # kg que quedan de la finca actual (por calibre)
    busy_hours_by_outlet: List[float]  # horas ocupadas acumuladas por máquina (para calcular utilización)
    bottleneck: str = ""  # descripción del cuello de botella actual
    capacity_last_period_kgph: float = 0.0  # kg/h efectivos en el último periodo
    buffer_capacity_by_caliber: Optional[Dict[Caliber, float]] = None  # capacidad por calibre
    blocked_hours: float = 0.0


@dataclass
class SimulationResult:
    """Resultado de una simulación."""

    seed: int
    total_packed_kg: float
    packed_kg_by_caliber: Dict[Caliber, float]
    packed_kg_by_outlet: List[float]  # kg embalados por cada máquina/salida
    utilization_by_outlet: List[float]
    idle_time_by_outlet_hours: List[float]
    blocked_time_infeed_hours: float
    max_queue_kg_by_caliber: Dict[Caliber, float]
