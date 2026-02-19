#!/usr/bin/env python3
"""
UI gráfica del simulador de empaque de limones.

Muestra dónde están los limones y cómo se mueven en el tiempo.
Ejecutar con: uv run streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Agregar src al path
base = Path(__file__).parent
sys.path.insert(0, str(base / "src"))

from lemon_packing.io.loaders import load_farms_from_csv, load_farms_from_dataframe, farms_to_dataframe, load_simulator_config
from lemon_packing.types import Assignment, Snapshot
from lemon_packing.sim.simpy_engine import run_simulation
from lemon_packing.sim.metrics import results_to_dataframe
from lemon_packing.opt import recommend_assignment

st.set_page_config(
    page_title="Simulador Empaque Limones",
    page_icon="🍋",
    layout="wide",
)

st.title("🍋 Simulador de Empaque de Limones")
st.caption("Visualización del flujo de limones: dónde están y cómo se mueven en el tiempo")

# Paths
config_path = base / "configs" / "simulator_config.yaml"
farms_csv = base / "data" / "farms.csv"

# Sidebar (siempre visible)
st.sidebar.header("Opciones")
run_live = st.sidebar.checkbox("Ejecutar simulación en vivo", value=True)
snapshot_interval = st.sidebar.slider("Intervalo de snapshots (min)", 5, 60, 15, 5)

# Inicializar session_state para overrides de configuración
if "config_overrides" not in st.session_state:
    st.session_state["config_overrides"] = None
if "farms_override" not in st.session_state:
    st.session_state["farms_override"] = None  # list[FarmLot] o None = usar CSV por defecto
if "farms_source" not in st.session_state:
    st.session_state["farms_source"] = "default"  # "default" | "upload" | "manual"

# Cargar config base desde YAML
config, extra = load_simulator_config(config_path)

# Pestañas: Configuración (para editar primero) y Simulación
tab_config, tab_sim = st.tabs(["⚙️ Configuración", "🍋 Simulación"])

# --- Pestaña Configuración ---
with tab_config:
    st.subheader("Parámetros editables (sin tocar el YAML)")

    # Valores actuales (overrides o YAML)
    overrides = st.session_state["config_overrides"] or {}
    caliber_by_outlet = overrides.get("caliber_by_outlet", extra["caliber_by_outlet"])
    arrival_rate = overrides.get("arrival_rate_kgph", extra["arrival_rate_kgph"])
    seed_base = overrides.get("seed_base", extra["seed_base"])
    shift_hours = overrides.get("shift_hours", config.shift_hours)
    snapshot_interval_default = extra.get("snapshot_interval_minutes") or 15

    with st.expander("🍋 Datos de fincas", expanded=True):
        st.caption("Elegí de dónde cargar los kg por calibre (80, 100, 120) de cada finca.")
        farms_source = st.radio(
            "Origen de datos",
            options=["default", "upload", "manual"],
            format_func=lambda x: {
                "default": "Archivo por defecto (data/farms.csv)",
                "upload": "Subir mi archivo CSV",
                "manual": "Cargar a mano",
            }[x],
            key="farms_source_radio",
            horizontal=True,
        )
        st.session_state["farms_source"] = farms_source

        if farms_source == "upload":
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
            elif st.session_state.get("farms_override") and "upload" in str(st.session_state.get("farms_source", "")):
                pass  # mantener override previo si ya había subido
            else:
                st.session_state["farms_override"] = None
                st.info("Subí un archivo CSV para usarlo.")

        elif farms_source == "manual":
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

        else:  # default
            st.session_state["farms_override"] = None
            try:
                default_farms = load_farms_from_csv(farms_csv)
                st.success(f"✓ Se usará el archivo por defecto ({len(default_farms)} fincas).")
            except Exception as e:
                st.error(f"No se pudo cargar el archivo por defecto: {e}")

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
        seed_base = st.number_input("Semilla base", min_value=0, value=int(seed_base), key="seed_base")
        shift_hours = st.number_input(
            "Duración del turno (h)",
            min_value=1.0,
            max_value=24.0,
            value=float(shift_hours),
            step=0.5,
            key="shift_hours",
        )

    col_apply, col_reset, _ = st.columns([1, 1, 3])
    with col_apply:
        if st.button("✅ Guardar y aplicar", type="primary"):
            st.session_state["config_overrides"] = {
                "caliber_by_outlet": new_caliber_by_outlet,
                "arrival_rate_kgph": arrival_rate,
                "seed_base": int(seed_base),
                "shift_hours": shift_hours,
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

# Aplicar overrides a config/extra
overrides = st.session_state["config_overrides"]
if overrides:
    extra = {**extra}
    if "caliber_by_outlet" in overrides:
        extra["caliber_by_outlet"] = overrides["caliber_by_outlet"]
    if "arrival_rate_kgph" in overrides:
        extra["arrival_rate_kgph"] = overrides["arrival_rate_kgph"]
    if "seed_base" in overrides:
        extra["seed_base"] = overrides["seed_base"]
    # PackingLineConfig es frozen, no podemos cambiar shift_hours fácilmente.
    # Por ahora lo dejamos - el shift_hours está en config y sería más trabajo modificar.
    # Si hace falta, habría que crear un nuevo PackingLineConfig.

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

    # Ejecutar simulación con captura de snapshots
    snapshots: list[Snapshot] = []
    config_obj = config
    assignment_obj = Assignment(caliber_by_outlet=extra["caliber_by_outlet"])

    # Optimizador: primeros 80 000 kg únicamente
    opt_result_first = recommend_assignment(config, farms, first_kg_limit=80000)

    if run_live:
        with st.spinner("Ejecutando simulación y capturando snapshots..."):
            r = run_simulation(
                config, farms, assignment_obj,
                extra["arrival_rate_kgph"],
                extra["seed_base"],
                snapshot_interval_minutes=snapshot_interval,
                snapshots_out=snapshots,
            )
            result_obj = r

    # --- Asignaciones: Tu asignación vs Optimizador primeros 80 000 kg ---
    st.header("📋 Asignación calibre-salida")

    total_kg = sum(sum(f.kg_by_caliber.values()) for f in farms)
    prop_esperadas = {
        c: (sum(f.kg_by_caliber.get(c, 0) for f in farms) / total_kg * 100) if total_kg > 0 else 0
        for c in sorted({c for f in farms for c in f.kg_by_caliber.keys()})
    }
    st.caption(f"Proporciones esperadas de fruta: " +
              ", ".join(f"c{c}={prop_esperadas[c]:.1f}%" for c in sorted(prop_esperadas)))

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Tu asignación")
        cap_tu = {c: 0 for c in prop_esperadas}
        for m, c in enumerate(assignment_obj.caliber_by_outlet):
            cap_tu[c] += config.speed_kgph[config.outlet_types[m]]
        total_cap = sum(cap_tu.values())
        for m, c in enumerate(assignment_obj.caliber_by_outlet):
            tipo = config.outlet_types[m]
            st.write(f"Salida #{m}: calibre {c} ({tipo})")
        if total_cap > 0:
            st.caption("Capacidad: " + ", ".join(
                f"c{c}={100*cap_tu[c]/total_cap:.1f}%" for c in sorted(cap_tu)))

    with cols[1]:
        st.subheader("Optimizador: primeros 80 000 kg")
        prop_first = opt_result_first.proportions_used or {}
        st.caption("Proporciones del tramo: " + ", ".join(
            f"c{c}={100*prop_first.get(c,0):.1f}%" for c in sorted(prop_first)))
        total_cap_first = sum(opt_result_first.capacity_by_caliber.values())
        for m, c in enumerate(opt_result_first.caliber_by_outlet):
            tipo = config.outlet_types[m]
            st.write(f"Salida #{m}: calibre {c} ({tipo})")
        if total_cap_first > 0:
            st.caption("Capacidad: " + ", ".join(
                f"c{c}={100*opt_result_first.capacity_by_caliber[c]/total_cap_first:.1f}%"
                for c in sorted(opt_result_first.capacity_by_caliber)))
        pct_max_first = (100 * opt_result_first.lambda_star / total_cap_first) if total_cap_first > 0 else 0
        st.metric("λ* sostenible", f"{opt_result_first.lambda_star:,.0f} kg/h", f"{pct_max_first:.1f}% del máximo")
        if st.button("Usar esta asignación", key="apply_opt_first"):
            prev = st.session_state.get("config_overrides") or {}
            st.session_state["config_overrides"] = {**prev, "caliber_by_outlet": list(opt_result_first.caliber_by_outlet)}
            st.rerun()

    if not snapshots:
        st.info(
            "Ejecutá la simulación con **Ejecutar simulación en vivo** activado para ver "
            "el flujo de limones en el tiempo."
        )
    else:
        # --- Slider de tiempo ---
        st.header("🕐 Momento en el turno")
        n_snap = len(snapshots)
        idx = st.slider(
            "Momento (snapshot)",
            min_value=0,
            max_value=n_snap - 1,
            value=0,
        )
        st.caption(f"Tiempo: **{snapshots[idx].t_hours:.1f} h**")

        snap = snapshots[idx]

        # Cuello de botella y capacidad del último periodo (por snapshot)
        bn_col1, bn_col2 = st.columns(2)
        with bn_col1:
            bottleneck = getattr(snap, "bottleneck", "") or "—"
            st.metric("Cuello de botella actual", bottleneck)
        with bn_col2:
            cap_last = getattr(snap, "capacity_last_period_kgph", 0) or 0
            max_cap = sum(config.speed_kgph[t] for t in config.outlet_types)
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
        max_cap_system = sum(config.speed_kgph[t] for t in config.outlet_types)
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
        )
        add_farm_change_vlines(fig_capacity, farm_change_times)
        fig_capacity.add_vline(x=snap.t_hours, line_dash="dash", line_color="red")
        st.plotly_chart(fig_capacity, use_container_width=True)

        # Cola en buffers por calibre
        fig_cola = go.Figure()
        for c in calibres:
            fig_cola.add_trace(go.Scatter(
                x=times,
                y=queue_df[f"Buffer c{c}"],
                name=f"Calibre {c}",
                fill="tozeroy",
                line=dict(width=2),
            ))
        fig_cola.update_layout(
            title="Limones en cola (buffers) — cómo se llenan y vacían",
            xaxis_title="Tiempo (h)",
            yaxis_title="kg en cola",
            height=350,
            hovermode="x unified",
        )
        add_farm_change_vlines(fig_cola, farm_change_times)
        st.plotly_chart(fig_cola, use_container_width=True)

        # Embalado acumulado por calibre
        fig_packed = go.Figure()
        for c in calibres:
            fig_packed.add_trace(go.Scatter(
                x=times,
                y=packed_df[f"Embalado c{c}"],
                name=f"Calibre {c}",
                mode="lines+markers",
                line=dict(width=2),
            ))
        fig_packed.update_layout(
            title="Embalado acumulado por calibre — cómo crece en el tiempo",
            xaxis_title="Tiempo (h)",
            yaxis_title="kg embalados",
            height=350,
            hovermode="x unified",
        )
        add_farm_change_vlines(fig_packed, farm_change_times)
        fig_packed.add_vline(x=snap.t_hours, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_packed, use_container_width=True)

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

        # --- Resumen final (expandible) ---
        with st.expander("Ver resumen numérico completo"):
            row = results_to_dataframe([result_obj]).iloc[0]
            packed_kg = row["total_packed_kg"]
            pct_procesado = (packed_kg / total_kg * 100) if total_kg > 0 else 0
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Procesados", f"{packed_kg:,.0f} kg", f"{pct_procesado:.1f}% del total")
            k2.metric("Kg total (fincas)", f"{total_kg:,.0f} kg")
            k3.metric("Tiempo bloqueo", f"{row['blocked_time_infeed_hours']:.2f} h")
            k4.metric("Seed", int(row["seed"]))
            st.dataframe(results_to_dataframe([result_obj]), use_container_width=True, hide_index=True)
