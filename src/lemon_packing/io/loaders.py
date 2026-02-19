"""Carga de configuración y escenarios desde archivos YAML y CSV."""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from lemon_packing.types import Assignment, FarmLot, PackingLineConfig


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Carga un archivo YAML como diccionario."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_simulator_config(path: str | Path = "configs/simulator_config.yaml") -> tuple[PackingLineConfig, Dict[str, Any]]:
    """
    Carga la configuración del simulador desde YAML.

    Returns:
        Tupla (PackingLineConfig, dict_extra) donde dict_extra contiene
        n_seeds, seed_base, arrival_rate_kgph para la simulación.
    """
    data = load_yaml(path)

    outlet_types = data["outlet_types"]
    speed_kgph = data["speed_kgph"]

    buffer_outlet = data.get("buffer_kg_by_outlet")
    if buffer_outlet is not None:
        buffer_kg_by_outlet = [float(b) for b in buffer_outlet]
    else:
        buffer_kg_by_outlet = None

    config = PackingLineConfig(
        shift_hours=float(data["shift_hours"]),
        dt_seconds=int(data["dt_seconds"]),
        outlet_types=outlet_types,
        speed_kgph=speed_kgph,
        buffer_kg_by_outlet=buffer_kg_by_outlet,
        total_buffer_blocking=data.get("total_buffer_blocking", True),
    )

    snapshot = data.get("snapshot_interval_minutes")
    opt_first = data.get("optimizer_first_kg")
    extra = {
        "arrival_rate_kgph": float(data.get("arrival_rate_kgph", 50000)),
        "n_seeds": int(data.get("n_seeds", 5)),
        "seed_base": int(data.get("seed_base", 42)),
        "caliber_by_outlet": [int(c) for c in data.get("caliber_by_outlet", [])],
        "snapshot_interval_minutes": int(snapshot) if snapshot is not None else None,
        "optimizer_first_kg": float(opt_first) if opt_first is not None else None,
    }

    return config, extra


def load_farms_from_csv(path: str | Path) -> List[FarmLot]:
    """
    Carga las fincas desde un CSV con 4 columnas:

    - finca: nombre de la finca
    - 80: kg de calibre 80
    - 100: kg de calibre 100
    - 120: kg de calibre 120

    Los nombres de las columnas de calibre deben ser exactamente "80", "100", "120".
    """
    df = pd.read_csv(path)
    return load_farms_from_dataframe(df)


def load_farms_from_dataframe(df: pd.DataFrame) -> List[FarmLot]:
    """
    Carga las fincas desde un DataFrame con columnas: finca, 80, 100, 120.
    Acepta columnas con espacios (se normalizan).
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    required = {"finca", "80", "100", "120"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}. "
            f"Columnas encontradas: {sorted(df.columns)}"
        )
    for c in ["80", "100", "120"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    farms = []
    for _, row in df.iterrows():
        kg_by_caliber = {
            80: float(row["80"]),
            100: float(row["100"]),
            120: float(row["120"]),
        }
        farm_id = str(row["finca"]).strip() if str(row["finca"]).strip() else f"F{len(farms)+1}"
        farms.append(FarmLot(farm_id=farm_id, kg_by_caliber=kg_by_caliber))
    return farms


def farms_to_dataframe(farms: List[FarmLot]) -> pd.DataFrame:
    """Convierte lista de FarmLot a DataFrame para edición."""
    calibers = sorted({c for f in farms for c in f.kg_by_caliber.keys()})
    if not calibers:
        calibers = [80, 100, 120]
    rows = []
    for f in farms:
        row = {"finca": f.farm_id}
        for c in calibers:
            row[str(c)] = f.kg_by_caliber.get(c, 0)
        rows.append(row)
    return pd.DataFrame(rows)


