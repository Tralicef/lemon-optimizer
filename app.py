#!/usr/bin/env python3
"""
UI gráfica del simulador de empaque de limones.

Muestra dónde están los limones y cómo se mueven en el tiempo.
Ejecutar con: uv run streamlit run app.py
"""

import sys
from pathlib import Path

import hashlib
import io
import json
import yaml

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.column_config import NumberColumn

# Agregar src al path
base = Path(__file__).parent
sys.path.insert(0, str(base / "src"))

from lemon_packing.io.loaders import load_farms_from_csv, load_farms_from_dataframe, farms_to_dataframe, load_simulator_config
from lemon_packing.types import Assignment, PackingLineConfig, Snapshot
from lemon_packing.sim.simpy_engine import run_simulation
from lemon_packing.sim.metrics import results_to_dataframe
from lemon_packing.opt import recommend_assignment
from lemon_packing.scenarios import (
    ANALISIS_COMBINADO_LABEL,
    PREDEFINED_SCENARIOS,
    jensen_shannon_index,
    merge_scenarios_equal_weight,
)


def _sim_cache_key(farms, assignment, config, extra):
    """Hash de los inputs que determinan optimizadores y simulaciones."""
    farms_data = [(f.farm_id, dict(f.kg_by_caliber)) for f in farms]
    key = json.dumps({
        "farms": farms_data,
        "assignment": assignment.caliber_by_outlet,
        "arrival": extra["arrival_rate_kgph"],
        "seed": extra["seed_base"],
        "shift_hours": config.shift_hours,
        "buffer": config.buffer_kg_by_outlet,
    }, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


def _compute_composition_variability(farms, calibers, first_kg_limit=None):
    """
    Calcula variabilidad de composición por calibre entre fincas.
    - Si first_kg_limit=None: variabilidad total (std de proporciones entre fincas).
    - Si first_kg_limit definido: diferencia entre mix del tramo inicial vs mix global.
    Returns: (variabilidad_total, variabilidad_primeros_n o None)
    """
    if not farms or not calibers:
        return 0.0, None
    # Proporciones por finca
    props_per_farm = []
    for f in farms:
        total = sum(f.kg_by_caliber.values())
        if total <= 0:
            continue
        props_per_farm.append({c: f.kg_by_caliber.get(c, 0) / total for c in calibers})
    if not props_per_farm:
        return 0.0, None
    # Variabilidad total: promedio de std por calibre
    variabilidad_total = 0.0
    for c in calibers:
        vals = [p.get(c, 0) for p in props_per_farm]
        variabilidad_total += np.std(vals) if len(vals) > 1 else 0.0
    variabilidad_total /= len(calibers) if calibers else 1

    # Proporciones globales
    total_kg = sum(sum(f.kg_by_caliber.values()) for f in farms)
    p_global = {c: sum(f.kg_by_caliber.get(c, 0) for f in farms) / total_kg if total_kg > 0 else 0 for c in calibers}

    if first_kg_limit is None:
        return variabilidad_total, None

    # Proporciones primeros N kg (mismo cálculo que el optimizador)
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
        return variabilidad_total, 0.0
    p_first = {c: kg_by_c[c] / total_taken for c in calibers}
    # Distancia L1 entre mix inicial y global
    variabilidad_first = sum(abs(p_first.get(c, 0) - p_global.get(c, 0)) for c in calibers)
    return variabilidad_total, variabilidad_first


def _margen_contribucion_kg(price_kg: float, costo_var_kg: float, margen_supuesto_por_kg: float) -> float:
    """Margen incremental por kg: precio − CV si hay precio; si no, margen supuesto por kg (ej. USD/18kg ÷ 18)."""
    if price_kg > 0:
        return float(price_kg) - float(costo_var_kg)
    return float(margen_supuesto_por_kg)


def _annual_margin_usd(dkg: float, margen_per_kg: float, shifts_year: float) -> float:
    """Margen incremental anual (USD) = margen/kg × Δ kg × turnos efectivos/año."""
    if shifts_year <= 0:
        return 0.0
    return float(dkg) * float(margen_per_kg) * float(shifts_year)


def _kg_tope_por_turno_oferta(campaign_kg_max: float, turnos_efectivos: float) -> float | None:
    """kg máx por turno si la oferta total de campaña está acotada; None = sin tope."""
    if campaign_kg_max <= 0 or turnos_efectivos <= 1e-9:
        return None
    return float(campaign_kg_max) / float(turnos_efectivos)


def _apply_supply_cap_kg(kg: float, kg_por_turno_max: float | None) -> float:
    """Acota kg procesados por turno cuando el cuello de botella es la oferta total de campaña."""
    if kg_por_turno_max is None:
        return float(kg)
    return min(float(kg), float(kg_por_turno_max))


def _payback_years_simple(capex: float, annual_net: float):
    """Años para recuperar CAPEX con flujo anual constante (sin descuento). None si no aplica."""
    if annual_net <= 1e-9 or capex <= 0:
        return None
    return float(capex) / float(annual_net)


def _npv_capex_annuity(capex: float, annual_net: float, n_years: int, r_annual: float) -> float:
    """VPN = −CAPEX + valor presente de anualidad constante de n años (flujos al cierre de cada año)."""
    n_years = int(max(0, n_years))
    capex = float(capex)
    a = float(annual_net)
    r = float(r_annual)
    if n_years <= 0:
        return -capex
    if r <= 1e-12:
        return -capex + a * n_years
    pv_inflows = a * (1.0 - (1.0 + r) ** (-n_years)) / r
    return -capex + pv_inflows


def _discounted_payback_years(capex: float, annual_net: float, r_annual: float, max_years: int = 500) -> float | None:
    """Años hasta que el VPN acumulado (flujos descontados a t0) sea ≥ 0; fracción dentro del año que cruza."""
    if annual_net <= 1e-9 or capex <= 0:
        return None
    r = float(r_annual)
    a = float(annual_net)
    cum = -float(capex)
    for y in range(1, max_years + 1):
        disc = a / ((1.0 + r) ** y)
        prev_cum = cum
        cum += disc
        if cum >= -1e-9:
            if disc > 1e-18 and prev_cum < -1e-9:
                frac = -prev_cum / disc
                frac = max(0.0, min(1.0, frac))
            else:
                frac = 1.0 if prev_cum < 0 else 0.0
            return float(y - 1) + frac
    return None


def _cashflow_rows_discounted(capex: float, annual_net: float, n_years_after_t0: int, r_annual: float):
    """Filas t0 + años: flujo nominal, factor 1/(1+r)^t, flujo actualizado y VPN acumulado."""
    rows = []
    r = float(r_annual)
    cum_npv = 0.0
    capex = float(capex)
    a = float(annual_net)
    df0 = 1.0
    disc0 = -capex * df0
    cum_npv += disc0
    rows.append(
        {
            "Período": "t0 (inversión)",
            "Flujo nominal (USD)": round(-capex, 2),
            "Factor desc.": round(df0, 6),
            "Flujo actualizado (USD)": round(disc0, 2),
            "VPN acumulado (USD)": round(cum_npv, 2),
        }
    )
    for y in range(1, n_years_after_t0 + 1):
        df = 1.0 / ((1.0 + r) ** y)
        disc = a * df
        cum_npv += disc
        rows.append(
            {
                "Período": f"Año {y}",
                "Flujo nominal (USD)": round(a, 2),
                "Factor desc.": round(df, 6),
                "Flujo actualizado (USD)": round(disc, 2),
                "VPN acumulado (USD)": round(cum_npv, 2),
            }
        )
    return rows


def _plotly_usd_axis(fig, *, secondary_y: bool = False):
    """Eje Y con prefijo $ y separador de miles."""
    if secondary_y:
        fig.update_yaxes(tickprefix="$", separatethousands=True, secondary_y=True)
    else:
        fig.update_yaxes(tickprefix="$", separatethousands=True)


def _st_df_money(df: pd.DataFrame):
    """Columnas numéricas con formato USD o entero según el nombre."""
    cfg = {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cl = c.lower()
        if c == "JSD":
            cfg[c] = NumberColumn(c, format="%.3f")
            continue
        if "breakeven" in cl or "%" in c:
            continue
        if "Δ kg" in c or "capacidad buffer" in cl:
            cfg[c] = NumberColumn(c, format="%.0f")
        elif "factor desc" in cl:
            cfg[c] = NumberColumn(c, format="%.6f")
        elif any(
            x in c
            for x in (
                "(USD)",
                "USD/",
                "CAPEX",
                "Margen/turno",
                "Margen anual",
                "Neto",
                "Amort",
                "Flujo",
                "Acumulado",
                "VPN",
                "Ingreso",
                "Balance",
                "Total ",
                "Costo USD",
                "CV empleado",
                "CV optim",
                "CV extra",
                "CV base",
                "CV c/",
                "CF turno",
                "Ganancia",
                "Margen extra",
                "Margen contrib",
                "Ingreso bruto",
            )
        ):
            cfg[c] = NumberColumn(c, format="$%.2f")
    return cfg


# Paleta de colores para gráficos (evitar blanco y negro en export HTML)
CALIBER_COLORS = {80: "#2E7D32", 100: "#1976D2", 120: "#D32F2F"}  # verde, azul, rojo
OUTLET_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7",
]

st.set_page_config(
    page_title="Simulador Empaque Limones",
    page_icon="🍋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🍋 Simulador de Empaque de Limones")
st.caption("Visualización del flujo de limones: dónde están y cómo se mueven en el tiempo")

# Paths
config_path = base / "configs" / "simulator_config.yaml"
farms_csv = base / "data" / "farms.csv"

run_live = True  # Simulación siempre activada
SNAPSHOT_INTERVAL_MIN = 5  # Fijo: cada 5 minutos

# Inicializar session_state para overrides de configuración
if "config_overrides" not in st.session_state:
    st.session_state["config_overrides"] = None
if "farms_override" not in st.session_state:
    st.session_state["farms_override"] = None  # list[FarmLot] o None = usar CSV por defecto
if "farms_source" not in st.session_state:
    st.session_state["farms_source"] = "default"  # "upload" | "manual" (solo cuando custom)
if "farms_scenario" not in st.session_state:
    st.session_state["farms_scenario"] = "muy_bajo"  # muy_bajo | bajo | medio | alto | custom

# Cargar config base desde YAML
config, extra = load_simulator_config(config_path)

# Pestañas: Configuración, Simulación y Análisis
tab_config, tab_sim, tab_analisis = st.tabs(["⚙️ Configuración", "🍋 Simulación", "📊 Análisis"])

# --- Pestaña Configuración ---
with tab_config:
    st.subheader("Parámetros editables (sin tocar el YAML)")

    # Valores actuales (overrides o YAML)
    overrides = st.session_state["config_overrides"] or {}
    caliber_by_outlet = overrides.get("caliber_by_outlet", extra["caliber_by_outlet"])
    arrival_rate = overrides.get("arrival_rate_kgph", extra["arrival_rate_kgph"])
    shift_hours = overrides.get("shift_hours", config.shift_hours)

    with st.expander("🍋 Datos de fincas", expanded=True):
        st.caption("Elegí un escenario predefinido (por índice Jensen-Shannon) o armá uno propio.")
        scenario_options = [
            (s.id, f"{s.name} — JSD {s.jensen_shannon:.3f}") for s in PREDEFINED_SCENARIOS
        ] + [("custom", "Personalizado (subir CSV o manual)")]
        scenario_labels = [opt[1] for opt in scenario_options]
        scenario_ids = [opt[0] for opt in scenario_options]
        sel_idx = scenario_ids.index(st.session_state.get("farms_scenario", "muy_bajo"))
        sel_label = st.selectbox(
            "Escenario",
            options=range(len(scenario_options)),
            format_func=lambda i: scenario_labels[i],
            index=sel_idx,
            key="scenario_select",
        )
        farms_scenario = scenario_ids[sel_label]
        st.session_state["farms_scenario"] = farms_scenario

        if farms_scenario == "custom":
            farms_source = st.radio(
                "Origen personalizado",
                options=["upload", "manual"],
                format_func=lambda x: {"upload": "Subir mi archivo CSV", "manual": "Cargar a mano"}[x],
                key="farms_source_radio",
                horizontal=True,
            )
            st.session_state["farms_source"] = farms_source
            # farms_override se setea en los bloques upload/manual cuando hay datos
        else:
            st.session_state["farms_source"] = None
            scenario = next(s for s in PREDEFINED_SCENARIOS if s.id == farms_scenario)
            st.session_state["farms_override"] = scenario.farms
            st.success(f"✓ Escenario **{scenario.name}**: {len(scenario.farms)} fincas, JSD = {scenario.jensen_shannon:.3f}")
            st.caption(scenario.description)

        if farms_scenario == "custom" and st.session_state.get("farms_source") == "upload":
            uploaded = st.file_uploader(
                "Subí un CSV con columnas: finca, 80, 100, 120",
                type=["csv"],
                key="farms_upload",
            )
            if uploaded is not None:
                try:
                    df_up = pd.read_csv(uploaded)
                    farms_from_upload = load_farms_from_dataframe(df_up)
                    st.session_state["farms_override"] = farms_from_upload
                    st.success(f"✓ Cargadas {len(farms_from_upload)} fincas desde tu archivo.")
                except Exception as e:
                    st.error(f"Error al leer el CSV: {e}")
                    if st.session_state.get("farms_override"):
                        del st.session_state["farms_override"]
            elif st.session_state.get("farms_override"):
                pass  # mantener override previo si ya había subido
            else:
                st.session_state["farms_override"] = None
                st.info("Subí un archivo CSV para usarlo.")

        elif farms_scenario == "custom" and st.session_state.get("farms_source") == "manual":
            # Cargar datos base: override previo (upload o manual) o CSV por defecto
            if st.session_state.get("farms_override"):
                default_df = farms_to_dataframe(st.session_state["farms_override"])
            else:
                try:
                    default_farms = load_farms_from_csv(farms_csv)
                    default_df = farms_to_dataframe(default_farms)
                except Exception:
                    default_df = pd.DataFrame({"finca": ["F1"], "80": [0], "100": [0], "120": [0]})

            st.caption("Editá la tabla y agregá o quitá filas. Columnas: finca (nombre), 80, 100, 120 (kg).")
            edited_df = st.data_editor(
                default_df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "finca": st.column_config.TextColumn("Finca", width="medium"),
                    "80": st.column_config.NumberColumn("Calibre 80 (kg)", min_value=0, step=100),
                    "100": st.column_config.NumberColumn("Calibre 100 (kg)", min_value=0, step=100),
                    "120": st.column_config.NumberColumn("Calibre 120 (kg)", min_value=0, step=100),
                },
                key="farms_data_editor",
            )
            if not edited_df.empty and "finca" in edited_df.columns:
                try:
                    farms_manual = load_farms_from_dataframe(edited_df)
                    st.session_state["farms_override"] = farms_manual
                    st.success(f"✓ {len(farms_manual)} fincas cargadas.")
                except Exception as e:
                    st.warning(f"Revisá los datos: {e}")
            else:
                st.session_state["farms_override"] = None

        elif farms_scenario == "custom":
            st.session_state["farms_override"] = None
            st.info("Elegí subir CSV o cargar a mano arriba.")

    with st.expander("📋 Asignación calibre por salida", expanded=True):
        st.caption("Cada salida procesa un calibre. Seleccioná 80, 100 o 120 para cada salida.")
        n_outlets = len(config.outlet_types)
        calibres_disponibles = [80, 100, 120]

        new_caliber_by_outlet = []
        cols_per_row = 4
        for i in range(0, n_outlets, cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(row_cols):
                idx = i + j
                if idx >= n_outlets:
                    break
                with col:
                    tipo = config.outlet_types[idx]
                    val_actual = caliber_by_outlet[idx]
                    nuevo = st.selectbox(
                        f"Salida #{idx} ({tipo})",
                        options=calibres_disponibles,
                        index=calibres_disponibles.index(val_actual),
                        key=f"caliber_outlet_{idx}",
                    )
                    new_caliber_by_outlet.append(nuevo)

    with st.expander("📊 Simulación y optimizador", expanded=True):
        arrival_rate = st.number_input(
            "Tasa de llegada (kg/h)",
            min_value=1000.0,
            max_value=100000.0,
            value=float(arrival_rate),
            step=1000.0,
            key="arrival_rate",
        )
        optimizer_first_kg = overrides.get("optimizer_first_kg", extra.get("optimizer_first_kg", 120000))
        optimizer_first_kg = st.number_input(
            "Optimizador: primeros N kg",
            min_value=1000,
            max_value=500000,
            value=int(optimizer_first_kg),
            step=10000,
            key="optimizer_first_kg",
        )
        shift_hours = st.number_input(
            "Duración del turno (h)",
            min_value=1.0,
            max_value=24.0,
            value=float(shift_hours),
            step=0.5,
            key="shift_hours",
        )
        buffer_default = config.buffer_kg_by_outlet[0] if config.buffer_kg_by_outlet else 190.0
        buffer_kg = overrides.get("buffer_kg_by_outlet", config.buffer_kg_by_outlet)
        buffer_kg_uniform = float(buffer_kg[0]) if buffer_kg and len(buffer_kg) > 0 else buffer_default
        buffer_kg_uniform = st.number_input(
            "Buffer kg por salida (todas)",
            min_value=10.0,
            max_value=2000.0,
            value=float(buffer_kg_uniform),
            step=10.0,
            key="buffer_kg",
        )

    col_apply, col_reset, _ = st.columns([1, 1, 3])
    with col_apply:
        if st.button("✅ Guardar y aplicar", type="primary"):
            n_outlets = len(config.outlet_types)
            buffer_list = [float(buffer_kg_uniform)] * n_outlets
            st.session_state["config_overrides"] = {
                "caliber_by_outlet": new_caliber_by_outlet,
                "arrival_rate_kgph": arrival_rate,
                "shift_hours": shift_hours,
                "optimizer_first_kg": optimizer_first_kg,
                "buffer_kg_by_outlet": buffer_list,
            }
            st.rerun()
    with col_reset:
        if st.button("↩️ Restaurar desde YAML"):
            st.session_state["config_overrides"] = None
            st.rerun()

    if st.session_state["config_overrides"]:
        st.success("Usando configuración personalizada (guardada arriba). La simulación se ejecutará con estos valores.")
    else:
        st.info("Usando valores del archivo simulator_config.yaml. Editá y guardá para sobreescribir.")

    # Descargas de configuración
    st.subheader("📥 Descargar configuración")
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        if st.session_state.get("farms_override"):
            farms_for_dl = st.session_state["farms_override"]
        else:
            try:
                farms_for_dl = load_farms_from_csv(farms_csv)
            except Exception:
                farms_for_dl = []
        if farms_for_dl:
            df_farms = farms_to_dataframe(farms_for_dl)
            csv_farms = df_farms.to_csv(index=False)
            st.download_button(
                "Descargar CSV fincas",
                data=csv_farms,
                file_name="farms.csv",
                mime="text/csv",
                key="dl_farms",
            )
        else:
            st.caption("Sin datos de fincas para descargar")
    with dl_col2:
        eff_caliber = overrides.get("caliber_by_outlet", extra["caliber_by_outlet"])
        df_asign = pd.DataFrame({
            "salida": range(len(eff_caliber)),
            "tipo": config.outlet_types,
            "calibre_asignado": eff_caliber,
        })
        csv_asign = df_asign.to_csv(index=False)
        st.download_button(
            "Descargar asignación máquinas",
            data=csv_asign,
            file_name="asignacion_maquinas.csv",
            mime="text/csv",
            key="dl_asign",
        )
    with dl_col3:
        config_yaml = {
            "shift_hours": overrides.get("shift_hours", config.shift_hours),
            "dt_seconds": config.dt_seconds,
            "arrival_rate_kgph": overrides.get("arrival_rate_kgph", extra["arrival_rate_kgph"]),
            "outlet_types": config.outlet_types,
            "speed_kgph": config.speed_kgph,
            "buffer_kg_by_outlet": overrides.get("buffer_kg_by_outlet", config.buffer_kg_by_outlet),
            "total_buffer_blocking": config.total_buffer_blocking,
            "caliber_by_outlet": overrides.get("caliber_by_outlet", extra["caliber_by_outlet"]),
            "optimizer_first_kg": overrides.get("optimizer_first_kg", extra.get("optimizer_first_kg")),
            "seed_base": extra.get("seed_base", 42),
        }
        yaml_str = yaml.dump(config_yaml, default_flow_style=False, allow_unicode=True, sort_keys=False)
        st.download_button(
            "Descargar config YAML",
            data=yaml_str,
            file_name="simulator_config.yaml",
            mime="text/yaml",
            key="dl_yaml",
        )

# Aplicar overrides a config/extra y construir config efectiva
overrides = st.session_state["config_overrides"]
config_obj = config
if overrides:
    extra = {**extra}
    if "caliber_by_outlet" in overrides:
        extra["caliber_by_outlet"] = overrides["caliber_by_outlet"]
    if "arrival_rate_kgph" in overrides:
        extra["arrival_rate_kgph"] = overrides["arrival_rate_kgph"]
    if "optimizer_first_kg" in overrides:
        extra["optimizer_first_kg"] = overrides["optimizer_first_kg"]
    # Construir config efectiva si hay overrides de shift_hours o buffer
    if "shift_hours" in overrides or "buffer_kg_by_outlet" in overrides:
        config_obj = PackingLineConfig(
            shift_hours=overrides.get("shift_hours", config.shift_hours),
            dt_seconds=config.dt_seconds,
            outlet_types=config.outlet_types,
            speed_kgph=config.speed_kgph,
            buffer_kg_by_outlet=overrides.get("buffer_kg_by_outlet", config.buffer_kg_by_outlet),
            total_buffer_blocking=config.total_buffer_blocking,
        )

# --- Pestaña Simulación ---
with tab_sim:
    # Cargar fincas: override (upload/manual) o archivo por defecto
    if st.session_state.get("farms_override"):
        farms = st.session_state["farms_override"]
    else:
        try:
            farms = load_farms_from_csv(farms_csv)
        except Exception as e:
            st.error(f"No se pudieron cargar las fincas: {e}")
            st.stop()
    farms_obj = farms

    assignment_obj = Assignment(caliber_by_outlet=extra["caliber_by_outlet"])
    cache_key = _sim_cache_key(farms, assignment_obj, config_obj, extra) + f"_{run_live}"

    # Usar cache si los inputs no cambiaron (ej. solo se movió el slider de momento)
    cached = (
        st.session_state.get("sim_cache_key") == cache_key
        and st.session_state.get("opt_empleado") is not None
        and st.session_state.get("opt_milp") is not None
    )

    if cached:
        opt_empleado = st.session_state["opt_empleado"]
        opt_milp = st.session_state["opt_milp"]
        result_obj = st.session_state.get("result_obj")
        result_empleado = st.session_state.get("result_empleado")
        result_milp = st.session_state.get("result_milp")
        snapshots = st.session_state.get("snapshots", [])
    else:
        # Empleado actual: promedio de calibres, heurística (instantáneo)
        opt_empleado = recommend_assignment(
            config_obj, farms,
            first_kg_limit=extra.get("optimizer_first_kg", 120000),
        )
        # Optimizador automático: maximiza kg procesados con asignación óptima + simulación
        with st.spinner("Calculando optimizador automático (maximiza kg procesados)..."):
            opt_milp = recommend_assignment(
                config_obj, farms, first_kg_limit=None, force_milp=True,
                use_mincost_style=True,
                arrival_rate_kgph=extra["arrival_rate_kgph"], seed=extra["seed_base"],
            )

        snapshots = []
        result_obj = None
        result_empleado = None
        result_milp = None
        if run_live:
            seed = extra["seed_base"]
            with st.spinner("Ejecutando 3 simulaciones (misma seed) para comparar kg procesados..."):
                result_obj = run_simulation(
                    config_obj, farms, assignment_obj,
                    extra["arrival_rate_kgph"], seed,
                    snapshot_interval_minutes=SNAPSHOT_INTERVAL_MIN,
                    snapshots_out=snapshots,
                )
                result_empleado = run_simulation(
                    config_obj, farms, Assignment(caliber_by_outlet=opt_empleado.caliber_by_outlet),
                    extra["arrival_rate_kgph"], seed,
                )
                result_milp = run_simulation(
                    config_obj, farms, Assignment(caliber_by_outlet=opt_milp.caliber_by_outlet),
                    extra["arrival_rate_kgph"], seed,
                )

        # Guardar en cache para no recalcular al mover el slider
        st.session_state["sim_cache_key"] = cache_key
        st.session_state["opt_empleado"] = opt_empleado
        st.session_state["opt_milp"] = opt_milp
        st.session_state["result_obj"] = result_obj
        st.session_state["result_empleado"] = result_empleado
        st.session_state["result_milp"] = result_milp
        st.session_state["snapshots"] = snapshots

    opt_first_kg = extra.get("optimizer_first_kg", 120000)  # para variabilidad

    # --- Asignaciones: Tu asignación vs Optimizador primeros 80 000 kg ---
    st.header("📋 Asignación calibre-salida")

    total_kg = sum(sum(f.kg_by_caliber.values()) for f in farms)
    calibers = sorted({c for f in farms for c in f.kg_by_caliber.keys()})
    prop_esperadas = {
        c: (sum(f.kg_by_caliber.get(c, 0) for f in farms) / total_kg * 100) if total_kg > 0 else 0
        for c in calibers
    }
    st.caption(f"Proporciones esperadas de fruta: " +
              ", ".join(f"c{c}={prop_esperadas[c]:.1f}%" for c in sorted(prop_esperadas)))

    # Kg procesados y total (visible arriba)
    if run_live and result_obj is not None:
        packed_kg = result_obj.total_packed_kg
        pct_procesado = (packed_kg / total_kg * 100) if total_kg > 0 else 0
        m1, m2 = st.columns(2)
        m1.metric("Kg procesados", f"{packed_kg:,.0f} kg", f"{pct_procesado:.1f}% del total")
        m2.metric("Kg total (fincas)", f"{total_kg:,.0f} kg")

    # Variabilidad de composición
    var_total, var_first = _compute_composition_variability(farms, calibers, opt_first_kg)
    jsd_actual = jensen_shannon_index(farms, calibers)
    var_col1, var_col2, var_col3 = st.columns(3)
    with var_col1:
        st.caption(f"**Variabilidad composición (total):** {var_total:.3f} — desv. estándar de proporciones entre fincas (0 = todas iguales)")
    with var_col2:
        if var_first is not None:
            st.caption(f"**Variabilidad primeros {opt_first_kg:,.0f} kg vs total:** {var_first:.3f} — diferencia entre mix inicial y global")
    with var_col3:
        st.caption(f"**Índice Jensen-Shannon:** {jsd_actual:.3f} — diferencia entre composiciones de fincas (0 = iguales, 1 = máx. distintas)")

    cols = st.columns(3)
    with cols[0]:
        st.subheader("Tu asignación")
        cap_tu = {c: 0 for c in prop_esperadas}
        for m, c in enumerate(assignment_obj.caliber_by_outlet):
            cap_tu[c] += config_obj.speed_kgph[config_obj.outlet_types[m]]
        total_cap = sum(cap_tu.values())
        for m, c in enumerate(assignment_obj.caliber_by_outlet):
            tipo = config_obj.outlet_types[m]
            st.write(f"Salida #{m}: calibre {c} ({tipo})")
        if total_cap > 0:
            st.caption("Capacidad: " + ", ".join(
                f"c{c}={100*cap_tu[c]/total_cap:.1f}%" for c in sorted(cap_tu)))
        # Throughput real = resultado de simular con TU asignación
        if run_live and result_obj is not None:
            kg_tu = result_obj.total_packed_kg
            pct_tu = (100 * kg_tu / total_kg) if total_kg > 0 else 0
            st.metric("Kg procesados", f"{kg_tu:,.0f} kg", f"{pct_tu:.1f}% del total")
        st.caption("_Para pruebas_")

    with cols[1]:
        st.subheader("Empleado actual")
        prop_emp = opt_empleado.proportions_used or {}
        first_kg = extra.get("optimizer_first_kg", 120000)
        st.caption(f"Mix de demanda (primeros {first_kg:,.0f} kg): " + ", ".join(
            f"c{c}={100*prop_emp.get(c,0):.1f}%" for c in sorted(prop_emp)))
        for m, c in enumerate(opt_empleado.caliber_by_outlet):
            tipo = config_obj.outlet_types[m]
            st.write(f"Salida #{m}: calibre {c} ({tipo})")
        if run_live and result_empleado is not None:
            kg_emp = result_empleado.total_packed_kg
            pct_emp = (100 * kg_emp / total_kg) if total_kg > 0 else 0
            delta_emp = kg_emp - result_obj.total_packed_kg if result_obj else 0
            delta_str = f"{delta_emp:+,.0f} kg vs tu asignación" if result_obj else f"{pct_emp:.1f}% del total"
            st.metric("Kg procesados", f"{kg_emp:,.0f} kg", delta_str)
        st.caption(f"Match capacidad ↔ demanda de los **primeros {first_kg:,.0f} kg** (heurística).")
        if st.button("Usar esta asignación", key="apply_opt_empleado"):
            prev = st.session_state.get("config_overrides") or {}
            st.session_state["config_overrides"] = {**prev, "caliber_by_outlet": list(opt_empleado.caliber_by_outlet)}
            st.rerun()

    with cols[2]:
        st.subheader("Optimizador automático")
        prop_milp = opt_milp.proportions_used or {}
        st.caption("Proporciones (promedio): " + ", ".join(
            f"c{c}={100*prop_milp.get(c,0):.1f}%" for c in sorted(prop_milp)))
        for m, c in enumerate(opt_milp.caliber_by_outlet):
            tipo = config_obj.outlet_types[m]
            st.write(f"Salida #{m}: calibre {c} ({tipo})")
        if run_live and result_milp is not None:
            kg_milp = result_milp.total_packed_kg
            pct_milp = (100 * kg_milp / total_kg) if total_kg > 0 else 0
            delta_milp = kg_milp - result_obj.total_packed_kg if result_obj else 0
            delta_str = f"{delta_milp:+,.0f} kg vs tu asignación" if result_obj else f"{pct_milp:.1f}% del total"
            st.metric("Kg procesados", f"{kg_milp:,.0f} kg", delta_str)
        st.caption("**Programación lineal entera** — robusto para todas las fincas.")
        if st.button("Usar esta asignación", key="apply_opt_milp"):
            prev = st.session_state.get("config_overrides") or {}
            st.session_state["config_overrides"] = {**prev, "caliber_by_outlet": list(opt_milp.caliber_by_outlet)}
            st.rerun()

    if not snapshots:
        st.info("Esperando resultados de la simulación para ver el flujo de limones en el tiempo.")
    else:
        # --- Slider de tiempo ---
        st.header("🕐 Momento en el turno")
        n_snap = len(snapshots)
        max_minutes = (n_snap - 1) * SNAPSHOT_INTERVAL_MIN
        minutos_actual = st.slider(
            "Momento (minutos desde inicio)",
            min_value=0,
            max_value=max_minutes,
            value=0,
            step=SNAPSHOT_INTERVAL_MIN,
        )
        idx = minutos_actual // SNAPSHOT_INTERVAL_MIN
        st.caption(f"Tiempo: **{snapshots[idx].t_hours:.1f} h** ({minutos_actual} min)")

        snap = snapshots[idx]

        # Cuello de botella y capacidad del último periodo (por snapshot)
        bn_col1, bn_col2 = st.columns(2)
        with bn_col1:
            bottleneck = getattr(snap, "bottleneck", "") or "—"
            st.metric("Cuello de botella actual", bottleneck)
        with bn_col2:
            cap_last = getattr(snap, "capacity_last_period_kgph", 0) or 0
            max_cap = sum(config_obj.speed_kgph[t] for t in config_obj.outlet_types)
            pct_cap = (100 * cap_last / max_cap) if max_cap > 0 else 0
            st.metric("Capacidad del sistema (último periodo)", f"{cap_last:,.0f} kg/h", f"{pct_cap:.1f}% del máximo")
        calibres = sorted(snap.queue_kg_by_caliber.keys())

        # --- Vista principal: diagrama del flujo ---
        st.header("🍋 Dónde están los limones")

        # Finca actual
        st.subheader(f"Finca en proceso: **{snap.farm_id}**")

        # Lo que queda de la finca actual (kg y proporción)
        remaining = getattr(snap, "remaining_kg_by_caliber", None) or {}
        total_remain = sum(remaining.values())
        if total_remain > 0:
            st.caption("**Lo que queda de esta finca:**")
            rem_cols = st.columns(len(calibres))
            for i, c in enumerate(calibres):
                kg = remaining.get(c, 0)
                pct = (kg / total_remain * 100) if total_remain > 0 else 0
                with rem_cols[i]:
                    st.metric(f"Calibre {c}", f"{kg:,.0f} kg", f"{pct:.1f}%")
        else:
            st.caption("_Finca terminada o sin datos_")

        # Buffers: "tanques" de limones esperando
        st.subheader("Buffers (limones en cola por calibre)")

        col_b1, col_b2, col_b3 = st.columns(3)

        colors = {80: "#90EE90", 100: "#98FB98", 120: "#3CB371"}  # verdes por calibre
        for i, (col, c) in enumerate(zip([col_b1, col_b2, col_b3], calibres)):
            with col:
                kg = snap.queue_kg_by_caliber.get(c, 0)
                cap = snap.buffer_capacity_by_caliber.get(c) if snap.buffer_capacity_by_caliber else 99999
                if cap and cap < 1e9:
                    pct = min(100, (kg / cap * 100)) if cap > 0 else 0
                    st.markdown(f"**Calibre {c}** — {kg:,.0f} kg ({pct:.0f}% del buffer)")
                    st.progress(pct / 100)
                else:
                    max_kg = max((s.queue_kg_by_caliber.get(c, 0) for s in snapshots), default=1)
                    pct = (kg / max_kg * 100) if max_kg > 0 else 0
                    st.markdown(f"**Calibre {c}** — {kg:,.0f} kg en cola")
                    st.progress(min(1.0, pct / 100))

        # Embalado por máquina hasta este momento
        st.subheader("Embalado por máquina hasta este momento")

        n_outlets = len(snap.packed_kg_by_outlet)
        outlet_labels = [f"#{m} (c{assignment_obj.caliber_by_outlet[m]})" for m in range(n_outlets)]
        packed_m = [snap.packed_kg_by_outlet[m] for m in range(n_outlets)]

        fig_outlets = go.Figure(go.Bar(
            x=outlet_labels,
            y=packed_m,
            marker_color=[colors.get(assignment_obj.caliber_by_outlet[m], "#87CEEB") for m in range(n_outlets)],
            text=[f"{v:,.0f} kg" for v in packed_m],
            textposition="outside",
        ))
        fig_outlets.update_layout(
            title="kg embalados por cada salida",
            xaxis_title="Salida",
            yaxis_title="kg",
            height=300,
        )
        st.plotly_chart(fig_outlets, use_container_width=True)

        # Utilización en el último periodo
        st.subheader("Utilización en el último periodo")
        busy_now = getattr(snap, "busy_hours_by_outlet", None) or [0.0] * n_outlets
        if idx > 0:
            snap_prev = snapshots[idx - 1]
            busy_prev = getattr(snap_prev, "busy_hours_by_outlet", None) or [0.0] * n_outlets
            period_h = snap.t_hours - snap_prev.t_hours
            if period_h > 1e-6:
                util_last = [(busy_now[m] - busy_prev[m]) / period_h for m in range(n_outlets)]
                fig_util_last = go.Figure(go.Bar(
                    x=outlet_labels,
                    y=[u * 100 for u in util_last],
                    marker_color=[colors.get(assignment_obj.caliber_by_outlet[m], "#87CEEB") for m in range(n_outlets)],
                    text=[f"{u:.0%}" for u in util_last],
                    textposition="outside",
                ))
                fig_util_last.update_layout(
                    title=f"Utilización en el periodo anterior ({period_h*60:.0f} min)",
                    xaxis_title="Salida",
                    yaxis_title="Utilización %",
                    yaxis=dict(range=[0, 105]),
                    height=260,
                )
                st.plotly_chart(fig_util_last, use_container_width=True)
            else:
                st.caption("_Período muy corto_")
        else:
            st.caption("_Primer snapshot: sin período anterior_")

        # --- Gráficos de evolución en el tiempo ---
        st.header("📈 Evolución en el tiempo")

        # Convertir snapshots a DataFrame
        times = [s.t_hours for s in snapshots]
        farms_seq = [s.farm_id for s in snapshots]

        # Detectar instantes de cambio de finca (para marcar con líneas verticales)
        farm_change_times = []
        for i in range(1, len(farms_seq)):
            if farms_seq[i] != farms_seq[i - 1]:
                farm_change_times.append((times[i], farms_seq[i]))

        def add_farm_change_vlines(fig, times_list):
            """Añade líneas verticales punteadas en cada cambio de finca."""
            for t, farm_id in times_list:
                fig.add_vline(
                    x=t,
                    line_dash="dot",
                    line_color="rgba(128,128,128,0.7)",
                    line_width=1.5,
                    annotation_text=farm_id,
                    annotation_position="top",
                    annotation_font_size=9,
                )

        queue_df = pd.DataFrame({
            f"Buffer c{c}": [s.queue_kg_by_caliber.get(c, 0) for s in snapshots]
            for c in calibres
        })
        queue_df["Tiempo (h)"] = times

        packed_df = pd.DataFrame({
            f"Embalado c{c}": [s.packed_kg_by_caliber.get(c, 0) for s in snapshots]
            for c in calibres
        })
        packed_df["Tiempo (h)"] = times

        # Capacidad del sistema en cada snapshot
        capacity_kgph = [getattr(s, "capacity_last_period_kgph", 0) or 0 for s in snapshots]
        max_cap_system = sum(config_obj.speed_kgph[t] for t in config_obj.outlet_types)
        fig_capacity = go.Figure()
        fig_capacity.add_trace(go.Scatter(
            x=times,
            y=capacity_kgph,
            name="Capacidad (kg/h)",
            mode="lines+markers",
            line=dict(width=2, color="#2E7D32"),
            marker=dict(size=6),
            text=[f"{v:,.0f} kg/h ({100*v/max_cap_system:.1f}%)" if max_cap_system > 0 else f"{v:,.0f} kg/h" for v in capacity_kgph],
            hoverinfo="text",
        ))
        fig_capacity.add_hline(
            y=max_cap_system,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Máx. {max_cap_system:,.0f} kg/h (100%)",
        )
        fig_capacity.update_layout(
            title="Capacidad del sistema en el último periodo — evolución en el turno",
            xaxis_title="Tiempo (h)",
            yaxis_title="kg/h",
            height=300,
            hovermode="x unified",
            template="plotly",
        )
        add_farm_change_vlines(fig_capacity, farm_change_times)
        fig_capacity.add_vline(x=snap.t_hours, line_dash="dash", line_color="red")
        st.plotly_chart(fig_capacity, use_container_width=True)

        # Cola en buffers por calibre
        fig_cola = go.Figure()
        for c in calibres:
            color = CALIBER_COLORS.get(c, "#333")
            fig_cola.add_trace(go.Scatter(
                x=times,
                y=queue_df[f"Buffer c{c}"],
                name=f"Calibre {c}",
                fill="tozeroy",
                line=dict(width=2, color=color),
            ))
        fig_cola.update_layout(
            title="Limones en cola (buffers) — cómo se llenan y vacían",
            xaxis_title="Tiempo (h)",
            yaxis_title="kg en cola",
            height=350,
            hovermode="x unified",
            template="plotly",
        )
        add_farm_change_vlines(fig_cola, farm_change_times)
        st.plotly_chart(fig_cola, use_container_width=True)

        # Cola (buffer) por salida (cada salida ve el buffer de su calibre asignado)
        n_outlets = len(assignment_obj.caliber_by_outlet)
        queue_outlet_df = pd.DataFrame({
            f"Salida #{m} (c{assignment_obj.caliber_by_outlet[m]})": [
                s.queue_kg_by_caliber.get(assignment_obj.caliber_by_outlet[m], 0) for s in snapshots
            ]
            for m in range(n_outlets)
        })
        queue_outlet_df["Tiempo (h)"] = times
        fig_cola_outlet = go.Figure()
        for m in range(n_outlets):
            label = f"Salida #{m} (c{assignment_obj.caliber_by_outlet[m]})"
            fig_cola_outlet.add_trace(go.Scatter(
                x=times,
                y=queue_outlet_df[label],
                name=label,
                line=dict(width=1.5 if m < 8 else 1, color=OUTLET_PALETTE[m % len(OUTLET_PALETTE)]),
            ))
        fig_cola_outlet.update_layout(
            title="Cola (buffer) por salida — nivel del buffer del calibre asignado a cada salida",
            xaxis_title="Tiempo (h)",
            yaxis_title="kg en cola",
            height=350,
            hovermode="x unified",
            template="plotly",
        )
        add_farm_change_vlines(fig_cola_outlet, farm_change_times)
        st.plotly_chart(fig_cola_outlet, use_container_width=True)

        # Embalado acumulado por calibre
        fig_packed = go.Figure()
        for c in calibres:
            fig_packed.add_trace(go.Scatter(
                x=times,
                y=packed_df[f"Embalado c{c}"],
                name=f"Calibre {c}",
                mode="lines+markers",
                line=dict(width=2, color=CALIBER_COLORS.get(c, "#333")),
                marker=dict(size=6, color=CALIBER_COLORS.get(c, "#333")),
            ))
        fig_packed.update_layout(
            title="Embalado acumulado por calibre — cómo crece en el tiempo",
            xaxis_title="Tiempo (h)",
            yaxis_title="kg embalados",
            height=350,
            hovermode="x unified",
            template="plotly",
        )
        add_farm_change_vlines(fig_packed, farm_change_times)
        fig_packed.add_vline(x=snap.t_hours, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_packed, use_container_width=True)

        # Embalado acumulado por salida
        packed_outlet_df = pd.DataFrame({
            f"Salida #{m} (c{assignment_obj.caliber_by_outlet[m]})": [
                s.packed_kg_by_outlet[m] for s in snapshots
            ]
            for m in range(n_outlets)
        })
        packed_outlet_df["Tiempo (h)"] = times
        fig_packed_outlet = go.Figure()
        for m in range(n_outlets):
            label = f"Salida #{m} (c{assignment_obj.caliber_by_outlet[m]})"
            color = OUTLET_PALETTE[m % len(OUTLET_PALETTE)]
            fig_packed_outlet.add_trace(go.Scatter(
                x=times,
                y=packed_outlet_df[label],
                name=label,
                mode="lines+markers",
                line=dict(width=1.5 if m < 8 else 1, color=color),
                marker=dict(size=4, color=color),
            ))
        fig_packed_outlet.update_layout(
            title="Embalado acumulado por salida — evolución en el tiempo",
            xaxis_title="Tiempo (h)",
            yaxis_title="kg embalados",
            height=350,
            hovermode="x unified",
            template="plotly",
        )
        add_farm_change_vlines(fig_packed_outlet, farm_change_times)
        fig_packed_outlet.add_vline(x=snap.t_hours, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_packed_outlet, use_container_width=True)

        # Fincas procesadas: timeline con bandas horizontales gruesas (estilo Gantt)
        unique_farms = []
        seen = set()
        for f in farms_seq:
            if f not in seen:
                seen.add(f)
                unique_farms.append(f)
        farm_order = {f: i for i, f in enumerate(unique_farms)}
        farm_labels = unique_farms
        _farm_palette = [
            "#E0E0E0", "#FFE4B5", "#FFD700", "#FFA07A", "#98FB98",
            "#87CEEB", "#DDA0DD", "#F0E68C", "#E6E6FA", "#FFB6C1",
        ]
        farm_colors = {f: _farm_palette[i % len(_farm_palette)] for i, f in enumerate(unique_farms)}
        bar_height = 0.7

        fig_farm = go.Figure()
        prev_farm = None
        prev_t = None
        for t, f in zip(times, farms_seq):
            if prev_farm != f:
                if prev_farm is not None and prev_t is not None:
                    y0 = farm_order.get(prev_farm, 0) - bar_height / 2
                    fig_farm.add_shape(
                        type="rect",
                        x0=prev_t, x1=t, y0=y0, y1=y0 + bar_height,
                        xref="x", yref="y",
                        fillcolor=farm_colors.get(prev_farm, "#CCC"),
                        line=dict(width=0),
                        layer="below",
                    )
                prev_farm = f
                prev_t = t
        if prev_farm is not None and prev_t is not None:
            y0 = farm_order.get(prev_farm, 0) - bar_height / 2
            fig_farm.add_shape(
                type="rect",
                x0=prev_t, x1=times[-1], y0=y0, y1=y0 + bar_height,
                xref="x", yref="y",
                fillcolor=farm_colors.get(prev_farm, "#CCC"),
                line=dict(width=0),
                layer="below",
            )
        fig_farm.add_trace(go.Scatter(
            x=times, y=[farm_order.get(f, 0) for f in farms_seq],
            mode="lines", line=dict(width=0),
            hoverinfo="text",
            text=[f"{f} hasta {t:.1f}h" for f, t in zip(farms_seq, times)],
        ))
        fig_farm.update_layout(
            title="Finca siendo procesada en cada momento",
            xaxis_title="Tiempo (h)",
            template="plotly",
            yaxis=dict(
                tickvals=list(range(len(unique_farms))),
                ticktext=farm_labels,
                range=[-0.6, max(0.6, len(unique_farms) - 1 + 0.6)],
            ),
            height=max(220, 80 + 50 * len(unique_farms)),
            showlegend=False,
            margin=dict(t=40, b=40),
        )
        add_farm_change_vlines(fig_farm, farm_change_times)
        fig_farm.add_vline(x=snap.t_hours, line_dash="dash", line_color="red", line_width=2)
        st.plotly_chart(fig_farm, use_container_width=True)

        # --- Descargas de resultados ---
        st.subheader("📥 Descargar resultados")
        res_dl1, res_dl2, res_dl3 = st.columns(3)
        with res_dl1:
            df_res = results_to_dataframe([result_obj])
            params_row = {
                "arrival_rate_kgph": extra["arrival_rate_kgph"],
                "shift_hours": config_obj.shift_hours,
                "total_kg_fincas": total_kg,
                "caliber_by_outlet": str(assignment_obj.caliber_by_outlet),
            }
            for k, v in params_row.items():
                df_res[k] = v
            csv_res = df_res.to_csv(index=False)
            st.download_button(
                "Descargar resultados CSV",
                data=csv_res,
                file_name="resultados_simulacion.csv",
                mime="text/csv",
                key="dl_resultados",
            )
        with res_dl2:
            params_yaml = {
                "parametros": {
                    "arrival_rate_kgph": extra["arrival_rate_kgph"],
                    "shift_hours": config_obj.shift_hours,
                    "total_kg_fincas": total_kg,
                    "caliber_by_outlet": list(assignment_obj.caliber_by_outlet),
                },
                "resultados": {
                    "total_packed_kg": float(result_obj.total_packed_kg),
                    "blocked_time_infeed_hours": result_obj.blocked_time_infeed_hours,
                    "utilization_by_outlet": [float(u) for u in result_obj.utilization_by_outlet],
                },
            }
            params_str = yaml.dump(params_yaml, default_flow_style=False, allow_unicode=True, sort_keys=False)
            st.download_button(
                "Descargar parámetros y resultados (YAML)",
                data=params_str,
                file_name="parametros_resultados.yaml",
                mime="text/yaml",
                key="dl_params",
            )
        with res_dl3:
            html_parts = [
                "<html><head><meta charset='utf-8'><title>Gráficos Simulación</title></head><body>",
                "<h1>Gráficos de simulación</h1>",
                "<h2>Capacidad del sistema</h2>",
                fig_capacity.to_html(full_html=False, include_plotlyjs="cdn"),
                "<h2>Cola en buffers por calibre</h2>",
                fig_cola.to_html(full_html=False, include_plotlyjs=False),
                "<h2>Cola en buffers por salida</h2>",
                fig_cola_outlet.to_html(full_html=False, include_plotlyjs=False),
                "<h2>Embalado acumulado por calibre</h2>",
                fig_packed.to_html(full_html=False, include_plotlyjs=False),
                "<h2>Embalado acumulado por salida</h2>",
                fig_packed_outlet.to_html(full_html=False, include_plotlyjs=False),
                "<h2>Finca en proceso</h2>",
                fig_farm.to_html(full_html=False, include_plotlyjs=False),
                "</body></html>",
            ]
            html_full = "\n".join(html_parts)
            st.download_button(
                "Descargar gráficos (HTML)",
                data=html_full,
                file_name="graficos_simulacion.html",
                mime="text/html",
                key="dl_graficos",
            )

        # --- Resumen final (expandible) ---
        with st.expander("Ver resumen numérico completo"):
            row = results_to_dataframe([result_obj]).iloc[0]
            packed_kg = row["total_packed_kg"]
            pct_procesado = (packed_kg / total_kg * 100) if total_kg > 0 else 0
            k1, k2, k3 = st.columns(3)
            k1.metric("Procesados", f"{packed_kg:,.0f} kg", f"{pct_procesado:.1f}% del total")
            k2.metric("Kg total (fincas)", f"{total_kg:,.0f} kg")
            k3.metric("Tiempo bloqueo", f"{row['blocked_time_infeed_hours']:.2f} h")
            df_display = results_to_dataframe([result_obj]).drop(columns=["seed"], errors="ignore")
            st.dataframe(df_display, use_container_width=True, hide_index=True)

# --- Pestaña Análisis ---
with tab_analisis:
    st.header("📊 Análisis comparativo")
    st.caption(
        "Empleado vs optimizador **por escenario JSD**; +1 máquina **AUTO** o **BULK**; buffers. "
        "El **análisis económico (§4–§6)** usa además **un caso combinado** (4 escenarios con peso igual). "
        "Opcional: **tope de oferta total en campaña** (kg) repartido en turnos efectivos si el límite es la fruta, no el empaque."
    )

    seed = extra["seed_base"]
    arrival = extra["arrival_rate_kgph"]

    def _total_capacity_kgph(config):
        """Capacidad total del sistema en kg/h."""
        return sum(config.speed_kgph[t] for t in config.outlet_types)

    def _config_with_extra_outlet(base_config, outlet_type="AUTO", buffer_kg=190):
        """Config con una máquina adicional."""
        new_types = list(base_config.outlet_types) + [outlet_type]
        base_buffers = base_config.buffer_kg_by_outlet or [190] * len(base_config.outlet_types)
        new_buffers = list(base_buffers) + [buffer_kg]
        return PackingLineConfig(
            shift_hours=base_config.shift_hours,
            dt_seconds=base_config.dt_seconds,
            outlet_types=new_types,
            speed_kgph=base_config.speed_kgph,
            buffer_kg_by_outlet=new_buffers,
            total_buffer_blocking=base_config.total_buffer_blocking,
        )

    def _config_with_scaled_buffers(base_config, scale: float):
        """Config con buffers escalados (ej. 1.5 = +50%, 2 = +100%, 4 = +300%)."""
        base_buffers = base_config.buffer_kg_by_outlet or [190] * len(base_config.outlet_types)
        new_buffers = [b * scale for b in base_buffers]
        return PackingLineConfig(
            shift_hours=base_config.shift_hours,
            dt_seconds=base_config.dt_seconds,
            outlet_types=base_config.outlet_types,
            speed_kgph=base_config.speed_kgph,
            buffer_kg_by_outlet=new_buffers,
            total_buffer_blocking=base_config.total_buffer_blocking,
        )

    with st.expander("Parámetros económicos (USD)", expanded=True):
        st.caption(
            "Costo variable referido al empleado actual. Amortización lineal de máquinas y buffers. "
            "**FOB** por caja 18 kg (default 14 USD) se muestra en el análisis; suma a precio y costo por kg sin cambiar el margen. "
            "Si cargás precio de venta, el margen/kg = precio − variable empaque; si precio = 0, se usa el **margen supuesto** "
            "(USD **cada 18 kg** ÷ 18 → USD/kg; default 1 USD/18 kg)."
        )
        econ_usd_per_18kg = st.number_input(
            "Costo total referencia cada 18 kg (empleado)", min_value=0.0, value=3.70, step=0.05, format="%.2f",
            key="econ_usd_per_18kg",
            help="Se descompone en variable + fijo según el % siguiente.",
        )
        econ_pct_costo_variable = st.slider(
            "% del costo/kg que es variable (resto = fijo del turno)",
            min_value=50,
            max_value=95,
            value=80,
            key="econ_pct_costo_variable",
            help="El fijo se asigna al turno según kg de referencia (empleado o base); el kg marginal solo arrastra el variable.",
        )
        econ_auto_usd = st.number_input("Costo máquina AUTO (USD)", min_value=0.0, value=80000.0, step=1000.0, key="econ_auto_usd")
        econ_bulk_usd = st.number_input("Costo máquina BULK (USD)", min_value=0.0, value=55000.0, step=1000.0, key="econ_bulk_usd")
        econ_amort_years = st.number_input("Años de amortización", min_value=1, value=15, step=1, key="econ_amort_years")
        econ_shifts_year = st.number_input(
            "Turnos por año (referencia si la planta operara 12 meses)",
            min_value=1,
            value=250,
            step=1,
            key="econ_shifts_year",
            help="Se combina con «meses con empaque» para obtener turnos efectivos al año (ej. 250 × 6/12).",
        )
        econ_months_operating = st.number_input(
            "Meses con empaque al año",
            min_value=1,
            max_value=12,
            value=6,
            step=1,
            key="econ_months_operating",
            help="Solo esos meses generan turnos: turnos efectivos/año = turnos referencia × (meses/12). "
            "Afecta amortización por turno, margen anual y flujo/breakeven.",
        )
        _default_oferta_campana_kg = 180.0 * 150_000.0  # ~180 días × 150.000 kg/día
        econ_oferta_campana_kg_max = st.number_input(
            "Tope de oferta total en campaña (kg, 0 = sin límite)",
            min_value=0.0,
            value=_default_oferta_campana_kg,
            step=100_000.0,
            format="%.0f",
            key="econ_oferta_campana_kg_max",
            help="Oferta máxima de fruta en la campaña (orden típico: días de cosecha × kg/día a planta; "
            "default ≈ 180 × 150.000 = 27 M kg). "
            "Se reparte en turnos efectivos = turnos referencia × (meses empaque ÷ 12); "
            "cada kg por turno del análisis se acota a oferta ÷ turnos. "
            "0 = no se limita la oferta (solo capacidad de línea en la simulación).",
        )
        econ_price_kg = st.number_input(
            "Precio venta USD/kg (0 = usar margen supuesto abajo)", min_value=0.0, value=0.0, step=0.01, format="%.3f",
            key="econ_price_kg",
        )
        econ_margen_usd_per_18kg = st.number_input(
            "Margen de ganancia supuesto (USD cada 18 kg, si precio = 0)",
            min_value=0.0,
            value=1.0,
            step=0.05,
            format="%.3f",
            key="econ_margen_usd_per_18kg",
            help="Si no cargás precio de venta, el margen/kg = este valor ÷ 18 (ej. 1 USD/18 kg ≈ 0,0556 USD/kg).",
        )
        econ_fob_usd_per_18kg = st.number_input(
            "FOB (USD por caja de 18 kg)",
            min_value=0.0,
            value=14.0,
            step=0.5,
            format="%.2f",
            key="econ_fob_usd_per_18kg",
            help="Se prorratea a USD/kg y se muestra junto al costo de empaque y al precio; "
            "al sumar lo mismo a ambos lados el margen/kg (precio − variable empaque) no cambia.",
        )
        econ_usd_per_kg_buffer_cap = st.number_input(
            "Inversión buffer: USD por kg de capacidad **nueva** (toda la línea)",
            min_value=0.0,
            value=12.0,
            step=1.0,
            format="%.2f",
            key="econ_usd_per_kg_buffer_cap",
            help="Ej.: ampliar +50% en todas las salidas suma capacidad = suma(buffers base)×0,5. "
            "Orden típico 8–20 USD/kg (estructura, sensores, obra). Default 12 USD/kg.",
        )
        econ_horizon_cf = st.number_input(
            "Horizonte flujo de inversión (años después de t0)",
            min_value=1,
            max_value=40,
            value=15,
            step=1,
            key="econ_horizon_cf",
            help="Cantidad de períodos anuales a mostrar en el flujo (además de t0). También define el horizonte del VPN en la tabla.",
        )
        econ_discount_rate_pct = st.number_input(
            "Tasa de descuento anual (%, costo de oportunidad del capital)",
            min_value=0.0,
            max_value=50.0,
            value=7.0,
            step=0.5,
            format="%.1f",
            key="econ_discount_rate_pct",
            help="Se usa para valor presente neto (VPN), payback descontado y flujos actualizados. "
            "7% anual es un orden típico nominal para análisis de inversión (ajustá según WACC o política).",
        )

    run_analisis = st.button("🔄 Calcular análisis", key="run_analisis")
    if run_analisis:
        st.session_state.pop("analisis_results", None)

    if run_analisis:
        turnos_ef_analisis = float(econ_shifts_year) * float(econ_months_operating) / 12.0
        kg_tope_turno = _kg_tope_por_turno_oferta(float(econ_oferta_campana_kg_max), turnos_ef_analisis)
        oferta_campaña_kg = float(econ_oferta_campana_kg_max) if econ_oferta_campana_kg_max > 0 else 0.0

        def _cap_kg_sim(kg: float) -> float:
            return _apply_supply_cap_kg(kg, kg_tope_turno)

        buffer_scales = [(1.5, "+50%"), (2.0, "+100%"), (4.0, "+300%")]
        n_buf = len(buffer_scales)
        n_scen = len(PREDEFINED_SCENARIOS)
        T_steps = n_scen * (1 + 2 + n_buf)
        T_econ = 1 + 2 + n_buf
        T_total = T_steps + T_econ
        idx_prog = 0
        rows_comp = []
        progress = st.progress(0, text="Analizando escenarios...")

        for scenario in PREDEFINED_SCENARIOS:
            progress.progress((idx_prog + 0.5) / T_total, text=f"Empleado vs Optimizador: {scenario.name}...")
            opt_emp = recommend_assignment(
                config_obj, scenario.farms,
                first_kg_limit=extra.get("optimizer_first_kg", 120000),
            )
            r_emp = run_simulation(config_obj, scenario.farms, Assignment(opt_emp.caliber_by_outlet), arrival, seed)
            with st.spinner(f"Optimizador para {scenario.name}..."):
                opt_milp_s = recommend_assignment(
                    config_obj, scenario.farms, first_kg_limit=None, force_milp=True,
                    use_mincost_style=True,
                    arrival_rate_kgph=arrival, seed=seed, local_search=False,
                )
            r_milp = run_simulation(config_obj, scenario.farms, Assignment(opt_milp_s.caliber_by_outlet), arrival, seed)
            k_emp = _cap_kg_sim(r_emp.total_packed_kg)
            k_milp = _cap_kg_sim(r_milp.total_packed_kg)
            diff = k_milp - k_emp
            pct_mejora = (100 * diff / k_emp) if k_emp > 0 else 0
            rows_comp.append({
                "Escenario": scenario.name,
                "JSD": round(scenario.jensen_shannon, 3),
                "Kg Empleado": round(k_emp, 0),
                "Kg Optimizador": round(k_milp, 0),
                "Diferencia (kg)": round(diff, 0),
                "% mejora": round(pct_mejora, 2),
            })
            idx_prog += 1

        cap_total_18 = _total_capacity_kgph(config_obj)
        maq_by_tipo = {}
        for maq_tipo in ("AUTO", "BULK"):
            maq_velocidad = config_obj.speed_kgph.get(maq_tipo, 2280)
            pct_cap_agregada = 100 * maq_velocidad / cap_total_18 if cap_total_18 > 0 else 0
            config_19 = _config_with_extra_outlet(config_obj, outlet_type=maq_tipo)
            rows_maq = []
            for scenario in PREDEFINED_SCENARIOS:
                progress.progress((idx_prog + 0.5) / T_total, text=f"Optim. vs optim. +1 {maq_tipo}: {scenario.name}...")
                opt_18 = recommend_assignment(
                    config_obj, scenario.farms, first_kg_limit=None, force_milp=True,
                    use_mincost_style=True, arrival_rate_kgph=arrival, seed=seed,
                    local_search=False,
                )
                r_18 = run_simulation(config_obj, scenario.farms, Assignment(opt_18.caliber_by_outlet), arrival, seed)
                opt_19 = recommend_assignment(
                    config_19, scenario.farms, first_kg_limit=None, force_milp=True,
                    use_mincost_style=True, arrival_rate_kgph=arrival, seed=seed,
                    local_search=False,
                )
                r_19 = run_simulation(config_19, scenario.farms, Assignment(opt_19.caliber_by_outlet), arrival, seed)
                k18 = _cap_kg_sim(r_18.total_packed_kg)
                k19 = _cap_kg_sim(r_19.total_packed_kg)
                diff_maq = k19 - k18
                pct_maq = (100 * diff_maq / k18) if k18 > 0 else 0
                ratio_ef = (pct_maq / pct_cap_agregada) if pct_cap_agregada > 0 else 0
                rows_maq.append({
                    "Escenario": scenario.name,
                    "JSD": round(scenario.jensen_shannon, 3),
                    "Kg (18 máq)": round(k18, 0),
                    "Kg (19 máq)": round(k19, 0),
                    "Diferencia (kg)": round(diff_maq, 0),
                    "% mejora kg": round(pct_maq, 2),
                    "% cap. agregada": round(pct_cap_agregada, 1),
                    "Ratio mejora/cap": round(ratio_ef, 2),
                })
                idx_prog += 1
            maq_by_tipo[maq_tipo] = {
                "rows": rows_maq,
                "velocidad": maq_velocidad,
                "pct_cap_agregada": pct_cap_agregada,
            }

        rows_buf = []
        for scale, label in buffer_scales:
            config_buf = _config_with_scaled_buffers(config_obj, scale)
            for scenario in PREDEFINED_SCENARIOS:
                progress.progress((idx_prog + 0.5) / T_total, text=f"Buffers {label}: {scenario.name}...")
                opt_base = recommend_assignment(
                    config_obj, scenario.farms, first_kg_limit=None, force_milp=True,
                    use_mincost_style=True, arrival_rate_kgph=arrival, seed=seed,
                    local_search=False,
                )
                opt_buf = recommend_assignment(
                    config_buf, scenario.farms, first_kg_limit=None, force_milp=True,
                    use_mincost_style=True, arrival_rate_kgph=arrival, seed=seed,
                    local_search=False,
                )
                r_base = run_simulation(config_obj, scenario.farms, Assignment(opt_base.caliber_by_outlet), arrival, seed)
                r_buf = run_simulation(config_buf, scenario.farms, Assignment(opt_buf.caliber_by_outlet), arrival, seed)
                kb = _cap_kg_sim(r_base.total_packed_kg)
                kbuf = _cap_kg_sim(r_buf.total_packed_kg)
                diff_buf = kbuf - kb
                pct_buf = (100 * diff_buf / kb) if kb > 0 else 0
                rows_buf.append({
                    "Escenario": scenario.name,
                    "JSD": round(scenario.jensen_shannon, 3),
                    "Buffers": label,
                    "Kg (base)": round(kb, 0),
                    "Kg (buffers)": round(kbuf, 0),
                    "Diferencia (kg)": round(diff_buf, 0),
                    "% mejora": round(pct_buf, 2),
                })
                idx_prog += 1

        # --- Solo para §4–§6: combinado (4 escenarios, peso igual por finca) ---
        farms_comb = merge_scenarios_equal_weight(PREDEFINED_SCENARIOS)
        jsd_comb = round(jensen_shannon_index(farms_comb, [80, 100, 120]), 3)
        econ_comp_rows = []
        progress.progress((idx_prog + 0.5) / T_total, text="Económico (combinado): empleado vs optimizador...")
        opt_emp_c = recommend_assignment(
            config_obj, farms_comb,
            first_kg_limit=extra.get("optimizer_first_kg", 120000),
        )
        r_emp_c = run_simulation(config_obj, farms_comb, Assignment(opt_emp_c.caliber_by_outlet), arrival, seed)
        with st.spinner("Optimizador (combinado, económico)..."):
            opt_milp_c = recommend_assignment(
                config_obj, farms_comb, first_kg_limit=None, force_milp=True,
                use_mincost_style=True,
                arrival_rate_kgph=arrival, seed=seed, local_search=False,
            )
        r_milp_c = run_simulation(config_obj, farms_comb, Assignment(opt_milp_c.caliber_by_outlet), arrival, seed)
        k_emp_c = _cap_kg_sim(r_emp_c.total_packed_kg)
        k_milp_c = _cap_kg_sim(r_milp_c.total_packed_kg)
        diff_c = k_milp_c - k_emp_c
        pct_c = (100 * diff_c / k_emp_c) if k_emp_c > 0 else 0
        econ_comp_rows.append({
            "Escenario": ANALISIS_COMBINADO_LABEL,
            "JSD": jsd_comb,
            "Kg Empleado": round(k_emp_c, 0),
            "Kg Optimizador": round(k_milp_c, 0),
            "Diferencia (kg)": round(diff_c, 0),
            "% mejora": round(pct_c, 2),
        })
        idx_prog += 1

        econ_maq_by_tipo = {}
        for maq_tipo in ("AUTO", "BULK"):
            maq_velocidad = config_obj.speed_kgph.get(maq_tipo, 2280)
            pct_cap_agregada = 100 * maq_velocidad / cap_total_18 if cap_total_18 > 0 else 0
            config_19 = _config_with_extra_outlet(config_obj, outlet_type=maq_tipo)
            progress.progress((idx_prog + 0.5) / T_total, text=f"Económico (combinado): +1 {maq_tipo}...")
            opt_18 = recommend_assignment(
                config_obj, farms_comb, first_kg_limit=None, force_milp=True,
                use_mincost_style=True, arrival_rate_kgph=arrival, seed=seed,
                local_search=False,
            )
            r_18 = run_simulation(config_obj, farms_comb, Assignment(opt_18.caliber_by_outlet), arrival, seed)
            opt_19 = recommend_assignment(
                config_19, farms_comb, first_kg_limit=None, force_milp=True,
                use_mincost_style=True, arrival_rate_kgph=arrival, seed=seed,
                local_search=False,
            )
            r_19 = run_simulation(config_19, farms_comb, Assignment(opt_19.caliber_by_outlet), arrival, seed)
            k18e = _cap_kg_sim(r_18.total_packed_kg)
            k19e = _cap_kg_sim(r_19.total_packed_kg)
            diff_maq = k19e - k18e
            pct_maq = (100 * diff_maq / k18e) if k18e > 0 else 0
            ratio_ef = (pct_maq / pct_cap_agregada) if pct_cap_agregada > 0 else 0
            econ_maq_by_tipo[maq_tipo] = {
                "rows": [{
                    "Escenario": ANALISIS_COMBINADO_LABEL,
                    "JSD": jsd_comb,
                    "Kg (18 máq)": round(k18e, 0),
                    "Kg (19 máq)": round(k19e, 0),
                    "Diferencia (kg)": round(diff_maq, 0),
                    "% mejora kg": round(pct_maq, 2),
                    "% cap. agregada": round(pct_cap_agregada, 1),
                    "Ratio mejora/cap": round(ratio_ef, 2),
                }],
                "velocidad": maq_velocidad,
                "pct_cap_agregada": pct_cap_agregada,
            }
            idx_prog += 1

        econ_rows_buf = []
        for scale, label in buffer_scales:
            config_buf = _config_with_scaled_buffers(config_obj, scale)
            progress.progress((idx_prog + 0.5) / T_total, text=f"Económico (combinado): buffers {label}...")
            opt_base = recommend_assignment(
                config_obj, farms_comb, first_kg_limit=None, force_milp=True,
                use_mincost_style=True, arrival_rate_kgph=arrival, seed=seed,
                local_search=False,
            )
            opt_buf = recommend_assignment(
                config_buf, farms_comb, first_kg_limit=None, force_milp=True,
                use_mincost_style=True, arrival_rate_kgph=arrival, seed=seed,
                local_search=False,
            )
            r_base = run_simulation(config_obj, farms_comb, Assignment(opt_base.caliber_by_outlet), arrival, seed)
            r_buf = run_simulation(config_buf, farms_comb, Assignment(opt_buf.caliber_by_outlet), arrival, seed)
            kbe = _cap_kg_sim(r_base.total_packed_kg)
            kbue = _cap_kg_sim(r_buf.total_packed_kg)
            diff_buf = kbue - kbe
            pct_buf = (100 * diff_buf / kbe) if kbe > 0 else 0
            econ_rows_buf.append({
                "Escenario": ANALISIS_COMBINADO_LABEL,
                "JSD": jsd_comb,
                "Buffers": label,
                "Kg (base)": round(kbe, 0),
                "Kg (buffers)": round(kbue, 0),
                "Diferencia (kg)": round(diff_buf, 0),
                "% mejora": round(pct_buf, 2),
            })
            idx_prog += 1

        progress.progress(1.0, text="Listo.")
        st.session_state["analisis_results"] = {
            "comp": rows_comp,
            "maq": maq_by_tipo,
            "buf": rows_buf,
            "econ_combined": {
                "comp": econ_comp_rows,
                "maq": econ_maq_by_tipo,
                "buf": econ_rows_buf,
                "jsd": jsd_comb,
            },
            "kg_tope_turno": kg_tope_turno,
            "oferta_campaña_kg_max": oferta_campaña_kg,
            "turnos_efectivos_analisis": turnos_ef_analisis,
        }

    if st.session_state.get("analisis_results"):
        rows_comp = st.session_state["analisis_results"]["comp"]
        _maq_raw = st.session_state["analisis_results"]["maq"]
        if isinstance(_maq_raw, dict) and "AUTO" in _maq_raw:
            maq_by_tipo = _maq_raw
        else:
            # Compatibilidad sesiones viejas: un solo bloque AUTO
            maq_by_tipo = {
                "AUTO": {
                    "rows": _maq_raw,
                    "velocidad": st.session_state["analisis_results"].get("maq_velocidad", 2280),
                    "pct_cap_agregada": st.session_state["analisis_results"].get("pct_cap_agregada", 0),
                },
            }
        rows_buf = st.session_state["analisis_results"].get("buf", [])
        econ_combined = st.session_state["analisis_results"].get("econ_combined")
        if econ_combined:
            rows_comp_econ = econ_combined["comp"]
            maq_by_tipo_econ = econ_combined["maq"]
            rows_buf_econ = econ_combined["buf"]
        else:
            rows_comp_econ = rows_comp
            maq_by_tipo_econ = maq_by_tipo
            rows_buf_econ = rows_buf

        # --- Parámetros económicos derivados (§1 gráficos USD y §4–6) ---
        h_turno = float(config_obj.shift_hours or 8.0)
        costo_usd_por_kg = econ_usd_per_18kg / 18.0 if econ_usd_per_18kg > 0 else 0.0
        turnos_efectivos_por_año = float(econ_shifts_year) * (float(econ_months_operating) / 12.0)
        amort_auto_turno = (
            econ_auto_usd / float(econ_amort_years) / turnos_efectivos_por_año
            if econ_amort_years > 0 and turnos_efectivos_por_año > 0
            else 0.0
        )
        amort_bulk_turno = (
            econ_bulk_usd / float(econ_amort_years) / turnos_efectivos_por_año
            if econ_amort_years > 0 and turnos_efectivos_por_año > 0
            else 0.0
        )
        fv = econ_pct_costo_variable / 100.0
        ff = 1.0 - fv
        costo_total_kg = costo_usd_por_kg
        costo_var_kg = costo_total_kg * fv
        fob_per_kg = float(econ_fob_usd_per_18kg) / 18.0 if econ_fob_usd_per_18kg > 0 else 0.0
        margen_supuesto_por_kg = float(econ_margen_usd_per_18kg) / 18.0
        margen_kg = _margen_contribucion_kg(econ_price_kg, costo_var_kg, margen_supuesto_por_kg)

        st.subheader("1. Empleado actual vs Optimizador automático por escenario JSD")
        _ar_sup = st.session_state.get("analisis_results") or {}
        _kg_tope = _ar_sup.get("kg_tope_turno")
        _oferta_cap = _ar_sup.get("oferta_campaña_kg_max", 0) or 0
        _tef_an = _ar_sup.get("turnos_efectivos_analisis")
        st.caption(
            "Cuatro escenarios de variabilidad entre fincas (JSD). El optimizador maximiza kg procesados. "
            f"Los **kg** son **por turno** (**{h_turno:.1f} h**). "
            f"Las tablas **USD (§4–§6)** usan el **caso combinado** (peso igual en los 4 escenarios). "
            f"**Anualizado** (§6): **{turnos_efectivos_por_año:.1f}** turnos/año "
            f"({econ_months_operating} meses × {econ_shifts_year} ref. ÷ 12)."
        )
        _tef_show = _tef_an if _tef_an else turnos_efectivos_por_año
        if _kg_tope is not None:
            st.caption(
                f"**Tope de oferta (campaña):** {_oferta_cap:,.0f} kg totales ÷ **{_tef_show:.1f}** turnos efectivos "
                f"→ máx. **{_kg_tope:,.0f} kg/turno** en tablas y gráficos (si la simulación supera eso, se acota). "
                "Poné **0** en «Tope de oferta…» para desactivar."
            )
        df_comp = pd.DataFrame(rows_comp)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="Empleado", x=[r["Escenario"] for r in rows_comp], y=[r["Kg Empleado"] for r in rows_comp]))
        fig_comp.add_trace(go.Bar(name="Optimizador", x=[r["Escenario"] for r in rows_comp], y=[r["Kg Optimizador"] for r in rows_comp]))
        fig_comp.update_layout(barmode="group", title="Kg procesados por turno y escenario", xaxis_title="Escenario", yaxis_title="kg")
        st.plotly_chart(fig_comp, use_container_width=True)
        fig_diff = go.Figure(go.Bar(x=[r["Escenario"] for r in rows_comp], y=[r["% mejora"] for r in rows_comp]))
        fig_diff.update_layout(title="% mejora del Optimizador vs Empleado por escenario", xaxis_title="Escenario", yaxis_title="% mejora")
        st.plotly_chart(fig_diff, use_container_width=True)
        fig_dkg = go.Figure(go.Bar(
            x=[r["Escenario"] for r in rows_comp],
            y=[r["Diferencia (kg)"] for r in rows_comp],
            marker_color="#2E7D32",
            name="Δ kg",
        ))
        fig_dkg.update_layout(
            title="Δ kg extra del optimizador vs empleado (mismo turno)",
            xaxis_title="Escenario",
            yaxis_title="kg",
        )
        st.plotly_chart(fig_dkg, use_container_width=True)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=[r["Kg Empleado"] for r in rows_comp],
            y=[r["Kg Optimizador"] for r in rows_comp],
            mode="markers+text",
            text=[r["Escenario"] for r in rows_comp],
            textposition="top center",
            name="Escenarios",
        ))
        mx = max([r["Kg Optimizador"] for r in rows_comp] + [r["Kg Empleado"] for r in rows_comp], default=1.0)
        fig_scatter.add_trace(go.Scatter(
            x=[0, mx], y=[0, mx], mode="lines", name="y = x (igual kg)", line=dict(dash="dash", color="gray"),
        ))
        fig_scatter.update_layout(
            title="Kg procesados: empleado vs optimizador (cada punto = un escenario JSD)",
            xaxis_title="Kg empleado (turno)",
            yaxis_title="Kg optimizador (turno)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        if margen_kg > 0:
            gan_turnos = [float(r["Diferencia (kg)"]) * margen_kg for r in rows_comp]
            fig_gan = go.Figure(go.Bar(
                x=[r["Escenario"] for r in rows_comp],
                y=gan_turnos,
                marker_color="#1565C0",
                name="Ganancia USD/turno",
            ))
            fig_gan.update_layout(
                title="Ganancia marginal estimada del optimizador (USD por turno, Δ kg × margen/kg)",
                xaxis_title="Escenario",
                yaxis_title="USD",
            )
            _plotly_usd_axis(fig_gan)
            st.plotly_chart(fig_gan, use_container_width=True)
            fig_mkg = go.Figure(go.Bar(
                x=[r["Escenario"] for r in rows_comp],
                y=[margen_kg] * len(rows_comp),
                marker_color="#5C6BC0",
                name="Margen/kg",
            ))
            fig_mkg.update_layout(
                title="Margen de contribución usado (USD/kg; mismo supuesto económico)",
                xaxis_title="Escenario",
                yaxis_title="USD/kg",
            )
            _plotly_usd_axis(fig_mkg)
            st.plotly_chart(fig_mkg, use_container_width=True)
        # Gráfico: cómo varía la mejora con el JSD (eje X numérico)
        fig_jsd = go.Figure(go.Scatter(
            x=[r["JSD"] for r in rows_comp],
            y=[r["% mejora"] for r in rows_comp],
            mode="lines+markers+text",
            text=[r["Escenario"] for r in rows_comp],
            textposition="top center",
        ))
        fig_jsd.update_layout(
            title="% mejora Optimizador vs Empleado según índice Jensen-Shannon",
            xaxis_title="Índice JSD (variabilidad entre fincas)",
            yaxis_title="% mejora",
        )
        st.plotly_chart(fig_jsd, use_container_width=True)

        st.subheader("2. Agregar una máquina (optimizador vs optimizador)")
        st.caption(
            "Misma línea **18 máquinas**: asignación óptima. **19 máquinas**: misma lógica con un equipo extra del tipo indicado. "
            "Una fila por escenario JSD; el resumen **USD** de inversión está en §4 con el caso combinado."
        )
        for j, maq_tipo in enumerate(("AUTO", "BULK")):
            if maq_tipo not in maq_by_tipo:
                continue
            md = maq_by_tipo[maq_tipo]
            rows_maq = md["rows"]
            pct_cap = md["pct_cap_agregada"]
            v_maq = md["velocidad"]
            st.markdown(f"#### 2.{chr(ord('a') + j)}. +1 máquina **{maq_tipo}**")
            st.caption(
                f"Optimizador (18 máq) vs optimizador (+1 **{maq_tipo}**, {v_maq:.0f} kg/h). "
                f"Capacidad total +~{pct_cap:.1f}%."
            )
            st.dataframe(pd.DataFrame(rows_maq), use_container_width=True, hide_index=True)
            st.caption(
                "**Ratio mejora/cap**: mejora % kg / % capacidad agregada "
                "(1 = proporcional; por debajo de 1 = rendimientos decrecientes)."
            )
            fig_maq = go.Figure()
            fig_maq.add_trace(go.Bar(name="18 máq", x=[r["Escenario"] for r in rows_maq], y=[r["Kg (18 máq)"] for r in rows_maq]))
            fig_maq.add_trace(go.Bar(name="19 máq", x=[r["Escenario"] for r in rows_maq], y=[r["Kg (19 máq)"] for r in rows_maq]))
            fig_maq.update_layout(barmode="group", title=f"Kg: 18 vs 19 máquinas (+1 {maq_tipo})", xaxis_title="Escenario", yaxis_title="kg")
            st.plotly_chart(fig_maq, use_container_width=True, key=f"maq_kg_{maq_tipo}")
            fig_maq_pct = go.Figure(go.Bar(x=[r["Escenario"] for r in rows_maq], y=[r["% mejora kg"] for r in rows_maq]))
            fig_maq_pct.update_layout(title=f"% mejora kg (+1 {maq_tipo}, ~{pct_cap:.1f}% capacidad)", xaxis_title="Escenario", yaxis_title="% mejora")
            st.plotly_chart(fig_maq_pct, use_container_width=True, key=f"maq_pct_{maq_tipo}")
            fig_maq_ratio = go.Figure(go.Bar(x=[r["Escenario"] for r in rows_maq], y=[r["Ratio mejora/cap"] for r in rows_maq]))
            fig_maq_ratio.update_layout(
                title=f"Ratio % mejora kg / % capacidad agregada (+1 {maq_tipo})",
                xaxis_title="Escenario",
                yaxis_title="Ratio",
            )
            st.plotly_chart(fig_maq_ratio, use_container_width=True, key=f"maq_ratio_{maq_tipo}")
            fig_maq_jsd = go.Figure(go.Scatter(
                x=[r["JSD"] for r in rows_maq],
                y=[r["% mejora kg"] for r in rows_maq],
                mode="lines+markers+text",
                text=[r["Escenario"] for r in rows_maq],
                textposition="top center",
            ))
            fig_maq_jsd.update_layout(
                title=f"% mejora al agregar 1 {maq_tipo} según JSD",
                xaxis_title="Índice JSD (variabilidad entre fincas)",
                yaxis_title="% mejora",
            )
            st.plotly_chart(fig_maq_jsd, use_container_width=True, key=f"maq_jsd_{maq_tipo}")

        st.subheader("3. Impacto de aumentar el tamaño de los buffers")
        st.caption("Por escenario: optimizador con buffers base vs +50%, +100% y +300%.")
        if rows_buf:
            df_buf = pd.DataFrame(rows_buf)
            st.dataframe(df_buf, use_container_width=True, hide_index=True)
            escenarios = list(dict.fromkeys(r["Escenario"] for r in rows_buf))
            fig_buf = go.Figure()
            def _get(e, label, col):
                for r in rows_buf:
                    if r["Escenario"] == e and r["Buffers"] == label:
                        return r[col]
                return 0

            fig_buf.add_trace(go.Bar(name="Base", x=escenarios, y=[_get(e, "+50%", "Kg (base)") for e in escenarios]))
            for label in ["+50%", "+100%", "+300%"]:
                fig_buf.add_trace(go.Bar(name=f"Buffers {label}", x=escenarios, y=[_get(e, label, "Kg (buffers)") for e in escenarios]))
            fig_buf.update_layout(barmode="group", title="Kg procesados según tamaño de buffers", xaxis_title="Escenario", yaxis_title="kg")
            st.plotly_chart(fig_buf, use_container_width=True)
            fig_buf_pct = go.Figure()
            for label in ["+50%", "+100%", "+300%"]:
                fig_buf_pct.add_trace(go.Bar(name=label, x=escenarios, y=[_get(e, label, "% mejora") for e in escenarios]))
            fig_buf_pct.update_layout(title="% mejora al aumentar buffers por escenario", xaxis_title="Escenario", yaxis_title="% mejora")
            st.plotly_chart(fig_buf_pct, use_container_width=True)

        # --- 4. Análisis económico (USD) — solo datos del caso combinado (4 escenarios, peso igual) ---
        st.subheader("4. Análisis económico (USD)")
        st.caption(
            "**Base de kg:** simulación **combinada** (promedio por finca entre Muy bajo / Bajo / Medio / Alto, 180k kg). "
            "Las secciones 1–3 muestran desgloses por escenario; acá los **montos** son para ese único caso combinado."
        )
        _margen_txt = (
            f"**precio − variable empaque** = **{margen_kg:.4f}** USD/kg (igual que sin desglosar FOB)"
            if econ_price_kg > 0
            else f"**margen supuesto** = **{econ_margen_usd_per_18kg:.2f} USD/18 kg** ÷ 18 = **{margen_kg:.4f}** USD/kg"
        )
        st.caption(
            f"**Empaque:** costo total ref. **{costo_total_kg:.4f} USD/kg** → variable **{costo_var_kg:.4f}** ({econ_pct_costo_variable}%) + "
            f"fijo **{100 - econ_pct_costo_variable}%** al turno (referencia kg empleado). "
            f"**FOB:** **{econ_fob_usd_per_18kg:.2f} USD/caja 18 kg** → **{fob_per_kg:.4f} USD/kg** "
            f"(se suma al costo de empaque y al precio de venta; el **margen/kg** es {_margen_txt}). "
            f"Costo/hora = costo total turno ÷ {h_turno:.1f} h. "
            f"**Operación:** **{econ_months_operating}** meses/año → **{turnos_efectivos_por_año:.1f}** turnos efectivos/año "
            f"(referencia {econ_shifts_year} × {econ_months_operating}/12). "
            f"Amort. AUTO/turno ≈ **${amort_auto_turno:,.2f}**, BULK ≈ **${amort_bulk_turno:,.2f}** (misma cuota anual repartida en menos turnos). "
            "**Ganancia del optimizador vs empleado** = Δ kg del **turno** × margen/kg → **USD por turno**."
        )

        rows_econ_comp = []
        for r in rows_comp_econ:
            kg_e = float(r["Kg Empleado"])
            kg_o = float(r["Kg Optimizador"])
            cv_e = kg_e * costo_var_kg
            cf_turno = kg_e * costo_total_kg * ff
            total_e = cv_e + cf_turno
            cv_o = kg_o * costo_var_kg
            total_o = cv_o + cf_turno
            ch_e = total_e / h_turno if h_turno > 0 else 0.0
            ch_o = total_o / h_turno if h_turno > 0 else 0.0
            dkg = float(r["Diferencia (kg)"])
            row_ec = {
                "Escenario": r["Escenario"],
                "JSD": r["JSD"],
                "CV empleado (USD)": round(cv_e, 2),
                "CF turno (USD)": round(cf_turno, 2),
                "Total empleado (USD)": round(total_e, 2),
                "CV optimizador (USD)": round(cv_o, 2),
                "Total optimizador (USD)": round(total_o, 2),
                "Costo USD/h empleado": round(ch_e, 2),
                "Costo USD/h optimizador": round(ch_o, 2),
                "Δ kg": round(dkg, 0),
            }
            if margen_kg > 0:
                ganancia_marginal = dkg * margen_kg
                row_ec["Margen contrib. (USD/kg)"] = round(margen_kg, 4)
                row_ec["Ganancia margen optim. (USD/turno)"] = round(ganancia_marginal, 2)
            rows_econ_comp.append(row_ec)
        st.markdown(
            "**Empleado vs optimizador** — el **fijo del turno** es el mismo para ambos (prorrateo según kg empleado); "
            "los kg extra del optimizador solo suman **costo variable**. "
            "Montos en **USD por turno** (caso combinado; no es el promedio de las filas de §1)."
        )
        df_ec = pd.DataFrame(rows_econ_comp)
        st.dataframe(df_ec, use_container_width=True, hide_index=True, column_config=_st_df_money(df_ec))
        if margen_kg > 0:
            st.caption(
                "**Ganancia (USD/turno)** = Δ kg × **margen/kg**; el margen/kg es "
                f"{'precio − costo variable empaque' if econ_price_kg > 0 else f'margen supuesto ({econ_margen_usd_per_18kg:.2f} USD/18 kg ÷ 18)'} "
                "(el FOB suma igual a precio y costo, no altera ese diferencial)."
            )
        else:
            st.caption("Definí **precio** o un **margen supuesto** positivo para estimar la ganancia marginal del optimizador.")

        st.markdown(
            "**Inversión: +1 máquina (optimizador vs optimizador)** — el Δ kg solo arrastra costo **variable**; "
            "amortización según costo **AUTO** o **BULK** en parámetros."
        )
        for maq_tipo in ("AUTO", "BULK"):
            if maq_tipo not in maq_by_tipo_econ:
                continue
            rows_maq = maq_by_tipo_econ[maq_tipo]["rows"]
            maq_cost = econ_auto_usd if maq_tipo == "AUTO" else econ_bulk_usd
            amort_maq_turno = (
                maq_cost / float(econ_amort_years) / turnos_efectivos_por_año
                if econ_amort_years > 0 and turnos_efectivos_por_año > 0
                else 0.0
            )
            rows_econ_maq = []
            for r in rows_maq:
                dkg = float(r["Diferencia (kg)"])
                cv_extra = dkg * costo_var_kg
                row_em = {
                    "Escenario": r["Escenario"],
                    "JSD": r["JSD"],
                    "Δ kg (19 vs 18 máq)": round(dkg, 0),
                    "CV extra (solo variable) (USD)": round(cv_extra, 2),
                    f"Amort. +1 {maq_tipo} / turno (USD)": round(amort_maq_turno, 2),
                }
                if margen_kg > 0:
                    ganancia_marginal = dkg * margen_kg
                    mb = ganancia_marginal - amort_maq_turno
                    if econ_price_kg > 0:
                        row_em["Ingreso extra (USD)"] = round(dkg * econ_price_kg, 2)
                        row_em["Ingreso bruto c/ FOB (USD)"] = round(dkg * (econ_price_kg + fob_per_kg), 2)
                    else:
                        row_em["Ingreso extra aprox. (USD)"] = round(dkg * (costo_var_kg + margen_kg + fob_per_kg), 2)
                    row_em["Ganancia margen − amort. (USD)"] = round(mb, 2)
                rows_econ_maq.append(row_em)
            st.markdown(f"##### +1 máquina **{maq_tipo}** (CAPEX ref. {maq_cost:,.0f} USD)")
            df_maq = pd.DataFrame(rows_econ_maq)
            st.dataframe(df_maq, use_container_width=True, hide_index=True, column_config=_st_df_money(df_maq))
        if margen_kg > 0:
            st.caption(
                "Ganancia margen = Δ kg × margen/kg; balance neto = ganancia margen − amort. máquina. "
                "Si no hay precio, el ingreso aprox. = Δ kg × (variable + margen supuesto en USD/kg, desde USD/18 kg)."
            )
        else:
            st.caption(
                "Definí precio o margen supuesto positivo para ver si el margen extra cubre la amortización por turno."
            )

        rows_econ_buf = []
        if rows_buf_econ:
            buf_list = config_obj.buffer_kg_by_outlet or [190.0] * len(config_obj.outlet_types)
            cap_buffer_base_kg = float(sum(buf_list))
            scale_por_etiqueta = {"+50%": 1.5, "+100%": 2.0, "+300%": 4.0}
            for r in rows_buf_econ:
                kg_b = float(r["Kg (buffers)"])
                kg0 = float(r["Kg (base)"])
                dkg = float(r["Diferencia (kg)"])
                cf_buf = kg0 * costo_total_kg * ff
                cv0 = kg0 * costo_var_kg
                cv_b = kg_b * costo_var_kg
                tot0 = cv0 + cf_buf
                tot_b = cv_b + cf_buf
                lbl = r["Buffers"]
                scale = scale_por_etiqueta.get(lbl, 1.0)
                delta_cap_kg = cap_buffer_base_kg * max(0.0, scale - 1.0)
                capex_buf = delta_cap_kg * econ_usd_per_kg_buffer_cap
                amort_buf_turno = (
                    capex_buf / float(econ_amort_years) / turnos_efectivos_por_año
                    if econ_amort_years > 0 and turnos_efectivos_por_año > 0 and capex_buf > 0
                    else 0.0
                )
                row_b = {
                    "Escenario": r["Escenario"],
                    "Buffers": lbl,
                    "Δ capacidad buffer (kg)": round(delta_cap_kg, 0),
                    "CAPEX buffer (USD)": round(capex_buf, 0),
                    "Amort. buffer / turno (USD)": round(amort_buf_turno, 2),
                    "CV base (USD)": round(cv0, 2),
                    "CF turno ref. base (USD)": round(cf_buf, 2),
                    "Total base (USD)": round(tot0, 2),
                    "CV c/ buffers (USD)": round(cv_b, 2),
                    "Total c/ buffers (USD)": round(tot_b, 2),
                    "Δ kg": round(dkg, 0),
                }
                if margen_kg > 0:
                    ganancia_delta = dkg * margen_kg
                    row_b["Ganancia margen Δ (USD)"] = round(ganancia_delta, 2)
                    row_b["Balance − amort. buffer (USD)"] = round(ganancia_delta - amort_buf_turno, 2)
                rows_econ_buf.append(row_b)
            st.markdown(
                "**Buffers (optimizador vs optimizador)** — inversión proporcional a la **capacidad nueva** de buffer (kg en toda la línea) "
                f"a **{econ_usd_per_kg_buffer_cap:.2f} USD/kg**; amortización en **{econ_amort_years}** años y "
                f"**{turnos_efectivos_por_año:.1f}** turnos efectivos/año ({econ_months_operating} meses). "
                f"Capacidad base total ≈ **{cap_buffer_base_kg:,.0f} kg**. "
                f"Costos de turno con reparto **{econ_pct_costo_variable}% variable / {100 - econ_pct_costo_variable}% fijo** (fijo anclado al kg base optimizador)."
            )
            df_eb = pd.DataFrame(rows_econ_buf)
            st.dataframe(df_eb, use_container_width=True, hide_index=True, column_config=_st_df_money(df_eb))
            st.caption(
                "Ganancia del Δ kg usa solo costo variable; balance neto resta amortización de la inversión en buffers. "
                "CAPEX orientativo — ajustá USD/kg según presupuesto."
            )

        # --- 5. Resumen: ¿conviene invertir? ---
        st.subheader("5. Resumen: ¿conviene invertir?")
        st.caption(
            "Criterio: **Sí** si el margen o balance neto del turno es **> 0 USD** con los supuestos actuales. "
            "Basado en el **caso combinado** (misma base que §4). No incluye impuestos ni otros costos corporativos."
        )
        if econ_price_kg > 0 and econ_price_kg <= costo_var_kg:
            st.error(
                f"El precio ({econ_price_kg:.4f} USD/kg) no supera el costo variable/kg ({costo_var_kg:.4f}): "
                "no hay margen en kg extra; el veredicto sería **No** salvo revisar supuestos."
            )
        elif margen_kg <= 0:
            st.error(
                "Margen incremental por kg no positivo: revisá **precio** o **margen supuesto (USD cada 18 kg)**."
            )
        else:
            res_opt = []
            n_pos_opt = 0
            for r in rows_comp_econ:
                dkg = float(r["Diferencia (kg)"])
                gan = dkg * margen_kg
                ok = gan > 1e-6
                if ok:
                    n_pos_opt += 1
                res_opt.append({
                    "Escenario": r["Escenario"],
                    "¿Conviene optimizador?": "Sí" if ok else ("Neutro" if abs(gan) <= 1e-6 else "No"),
                    "Margen extra (USD/turno)": round(gan, 2),
                })
            st.markdown("**Optimizador vs empleado** (sin inversión en activos; solo mejor asignación; caso combinado)")
            df_ro = pd.DataFrame(res_opt)
            st.dataframe(df_ro, use_container_width=True, hide_index=True, column_config=_st_df_money(df_ro))
            st.success(
                f"**{n_pos_opt}/{len(rows_comp_econ)}** fila(s) con margen positivo (caso combinado)."
            )

            for maq_tipo in ("AUTO", "BULK"):
                if maq_tipo not in maq_by_tipo_econ:
                    continue
                rows_maq = maq_by_tipo_econ[maq_tipo]["rows"]
                maq_cost = econ_auto_usd if maq_tipo == "AUTO" else econ_bulk_usd
                amort_maq_turno = (
                    maq_cost / float(econ_amort_years) / turnos_efectivos_por_año
                    if econ_amort_years > 0 and turnos_efectivos_por_año > 0
                    else 0.0
                )
                res_maq = []
                n_pos_maq = 0
                for r in rows_maq:
                    dkg = float(r["Diferencia (kg)"])
                    gan = dkg * margen_kg
                    mb = gan - amort_maq_turno
                    ok = mb > 1e-6
                    if ok:
                        n_pos_maq += 1
                    res_maq.append({
                        "Escenario": r["Escenario"],
                        f"¿Conviene +1 {maq_tipo}?": "Sí" if ok else ("Dudoso" if mb > -5 else "No"),
                        "Neto vs amort. (USD/turno)": round(mb, 2),
                        "Amort. máquina/turno": round(amort_maq_turno, 2),
                    })
                st.markdown(
                    f"**Inversión: +1 máquina {maq_tipo}** (optimizador vs optimizador; CAPEX **{maq_cost:,.0f} USD**, amort. {econ_amort_years} a.)"
                )
                df_rm = pd.DataFrame(res_maq)
                st.dataframe(df_rm, use_container_width=True, hide_index=True, column_config=_st_df_money(df_rm))
                if n_pos_maq == len(rows_maq):
                    st.success(
                        f"**{maq_tipo}:** en el **caso combinado** el neto del turno supera la amortización."
                    )
                elif n_pos_maq > 0:
                    st.warning(
                        f"**{maq_tipo}:** solo **{n_pos_maq}/{len(rows_maq)}** fila(s) con neto positivo."
                    )
                else:
                    st.error(
                        f"**{maq_tipo}:** el margen extra no cubre la amortización con estos supuestos (caso combinado)."
                    )

            if rows_buf_econ and rows_econ_buf:
                res_buf = []
                n_pos_buf = 0
                for rb in rows_econ_buf:
                    bal = rb.get("Balance − amort. buffer (USD)")
                    if bal is None:
                        continue
                    ok = float(bal) > 1e-6
                    if ok:
                        n_pos_buf += 1
                    res_buf.append({
                        "Escenario": rb["Escenario"],
                        "Ampliación": rb["Buffers"],
                        "¿Conviene?": "Sí" if ok else ("Dudoso" if float(bal) > -5 else "No"),
                        "Neto vs amort. (USD/turno)": bal,
                    })
                st.markdown("**Inversión en buffers** (caso combinado; según CAPEX y amort. asumidos)")
                df_rb = pd.DataFrame(res_buf)
                st.dataframe(df_rb, use_container_width=True, hide_index=True, column_config=_st_df_money(df_rb))
                nbuf = len(res_buf)
                if nbuf and n_pos_buf == nbuf:
                    st.success("En **todos** los casos el neto supera la amortización de la ampliación de buffers.")
                elif n_pos_buf > 0:
                    st.warning(
                        f"**{n_pos_buf}/{nbuf}** combinaciones escenario × nivel de buffer con balance positivo."
                    )
                else:
                    st.error("Ninguna combinación muestra balance neto positivo con amortización de buffers.")

        # --- 6. Flujo de inversión (t0) y breakeven ---
        st.subheader("6. Flujo de inversión (t0) y breakeven")
        st.caption(
            "Misma base **combinada** que §4–§5. "
            "**t0:** desembolso de CAPEX (salida de caja). **Años 1…N:** ingreso anual = "
            "Δ kg × margen/kg × **turnos efectivos/año** "
            f"({turnos_efectivos_por_año:.1f} = {econ_shifts_year} turnos referencia × {econ_months_operating}/12 meses). "
            f"**Descuento:** **{econ_discount_rate_pct:.1f}%** anual sobre flujos al cierre de cada año (VPN y payback descontado). "
            "No incluye impuestos ni valor residual; la amortización contable no es flujo de caja."
        )
        h_cf = int(econ_horizon_cf)
        r_disc = float(econ_discount_rate_pct) / 100.0
        if econ_price_kg > 0 and econ_price_kg <= costo_var_kg:
            st.warning("Precio no supera costo variable: no hay flujo de recuperación con ese precio.")
        elif margen_kg <= 0:
            st.warning("Margen/kg no positivo: no hay flujos de recuperación.")
        else:
            vpn_col = f"VPN ({h_cf} a) (USD)"
            breakeven_rows = []
            for r in rows_comp_econ:
                dkg = float(r["Diferencia (kg)"])
                gan_turno = dkg * margen_kg
                ann = _annual_margin_usd(dkg, margen_kg, turnos_efectivos_por_año)
                npv_row = round(_npv_capex_annuity(0.0, ann, h_cf, r_disc), 2)
                breakeven_rows.append({
                    "Proyecto": f"OPT — {r['Escenario']}",
                    "Tipo": "Optimizador (sin CAPEX)",
                    "Escenario": r["Escenario"],
                    "CAPEX (USD)": 0.0,
                    "Margen/turno (USD)": round(gan_turno, 2),
                    "Margen anual (USD)": round(ann, 2),
                    vpn_col: npv_row,
                    "Breakeven (años)": "— (sin inversión)",
                    "Payback descont. (años)": "—",
                })
            for maq_tipo in ("AUTO", "BULK"):
                if maq_tipo not in maq_by_tipo_econ:
                    continue
                capex_m = float(econ_auto_usd if maq_tipo == "AUTO" else econ_bulk_usd)
                for r in maq_by_tipo_econ[maq_tipo]["rows"]:
                    dkg = float(r["Diferencia (kg)"])
                    gan_turno = dkg * margen_kg
                    ann = _annual_margin_usd(dkg, margen_kg, turnos_efectivos_por_año)
                    pb = _payback_years_simple(capex_m, ann)
                    pb_d = _discounted_payback_years(capex_m, ann, r_disc)
                    npv_row = round(_npv_capex_annuity(capex_m, ann, h_cf, r_disc), 2)
                    breakeven_rows.append({
                        "Proyecto": f"{maq_tipo} — {r['Escenario']}",
                        "Tipo": f"+1 máquina {maq_tipo}",
                        "Escenario": r["Escenario"],
                        "CAPEX (USD)": capex_m,
                        "Margen/turno (USD)": round(gan_turno, 2),
                        "Margen anual (USD)": round(ann, 2),
                        vpn_col: npv_row,
                        "Breakeven (años)": round(pb, 2) if pb is not None else "—",
                        "Payback descont. (años)": round(pb_d, 2) if pb_d is not None else "—",
                    })
            for rb in rows_econ_buf:
                capex_b = float(rb.get("CAPEX buffer (USD)", 0) or 0)
                dkg = float(rb["Δ kg"])
                gan_turno = dkg * margen_kg
                ann = _annual_margin_usd(dkg, margen_kg, turnos_efectivos_por_año)
                pb = _payback_years_simple(capex_b, ann)
                pb_d = _discounted_payback_years(capex_b, ann, r_disc)
                npv_row = round(_npv_capex_annuity(capex_b, ann, h_cf, r_disc), 2)
                breakeven_rows.append({
                    "Proyecto": f"BUF {rb['Buffers']} — {rb['Escenario']}",
                    "Tipo": f"Buffers {rb['Buffers']}",
                    "Escenario": rb["Escenario"],
                    "CAPEX (USD)": capex_b,
                    "Margen/turno (USD)": round(gan_turno, 2),
                    "Margen anual (USD)": round(ann, 2),
                    vpn_col: npv_row,
                    "Breakeven (años)": round(pb, 2) if pb is not None else "—",
                    "Payback descont. (años)": round(pb_d, 2) if pb_d is not None else "—",
                })

            df_br = pd.DataFrame(breakeven_rows)
            st.dataframe(df_br, use_container_width=True, hide_index=True, column_config=_st_df_money(df_br))
            st.caption(
                f"**Payback simple** = CAPEX ÷ margen anual (sin descuento). **VPN ({h_cf} a)** y **payback descontado** "
                f"usan **{econ_discount_rate_pct:.1f}%** anual. Con CAPEX = 0 el VPN es solo el valor presente del margen anual."
            )

            sel_cf = st.selectbox(
                "Detalle: flujo de caja y acumulado",
                options=[x["Proyecto"] for x in breakeven_rows],
                key="cf_select_inversion",
            )
            row_sel = next(x for x in breakeven_rows if x["Proyecto"] == sel_cf)
            capex_s = float(row_sel["CAPEX (USD)"])
            ann_s = float(row_sel["Margen anual (USD)"])
            tbl_cf = _cashflow_rows_discounted(capex_s, ann_s, h_cf, r_disc)
            df_cf = pd.DataFrame(tbl_cf)
            st.dataframe(df_cf, use_container_width=True, hide_index=True, column_config=_st_df_money(df_cf))
            periods = [t["Período"] for t in tbl_cf]
            flows = [t["Flujo nominal (USD)"] for t in tbl_cf]
            vpn_cum = [t["VPN acumulado (USD)"] for t in tbl_cf]
            fig_cf = go.Figure()
            fig_cf.add_trace(go.Bar(x=periods, y=flows, name="Flujo nominal (USD)"))
            fig_cf.add_trace(go.Scatter(x=periods, y=vpn_cum, name="VPN acumulado (USD)", mode="lines+markers"))
            fig_cf.update_yaxes(title_text="USD (nominal por período; línea = VPN acumulado al desc.)")
            _plotly_usd_axis(fig_cf, secondary_y=False)
            fig_cf.update_layout(
                title=f"Flujo de inversión: {sel_cf} (descuento {econ_discount_rate_pct:.1f}% anual)",
                xaxis_title="Período",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_cf, use_container_width=True, key="fig_cf_flujo_inversion")
            if capex_s > 1e-6 and ann_s > 1e-6:
                pb = _payback_years_simple(capex_s, ann_s)
                pb_d = _discounted_payback_years(capex_s, ann_s, r_disc)
                npv_sel = _npv_capex_annuity(capex_s, ann_s, h_cf, r_disc)
                m1, m2, m3 = st.columns(3)
                m1.metric("Payback simple (años)", f"{pb:.2f}" if pb is not None else "—")
                m2.metric("Payback descontado (años)", f"{pb_d:.2f}" if pb_d is not None else "—")
                m3.metric(f"VPN ({h_cf} años)", f"${npv_sel:,.0f}")
            elif capex_s <= 1e-6:
                npv_op = _npv_capex_annuity(0.0, ann_s, h_cf, r_disc)
                st.info(
                    f"Sin CAPEX en t0: el margen anual es beneficio recurrente directo. "
                    f"VPN ({h_cf} a) del margen: **${npv_op:,.0f}** al {econ_discount_rate_pct:.1f}% anual."
                )
            else:
                st.warning("Margen anual no positivo: no se alcanza recuperación del capital con estos supuestos.")

    else:
        st.info("Hacé clic en **Calcular análisis** para ejecutar los análisis.")
