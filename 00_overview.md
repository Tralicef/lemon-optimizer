# Optimización de Empaque de Limones por Calibre — Documentación técnica

Este paquete contiene documentación en Markdown (md) separada por componente:

- `00_overview.md`: visión general, alcance, supuestos, roadmap y artefactos.
- `01_simulador.md`: especificación completa del simulador de eventos discretos (SimPy), métricas, interfaces y skeleton de código.
- `02_optimizador_v1.md`: formulación y código base de un optimizador v1 (asignación estática) + validación con simulación.
- `03_optimizador_v2.md`: formulación y enfoque de optimizador v2 (cambios de calibre con setup time) usando OR-Tools CP-SAT + extensiones.

## Supuestos (base)
- La fruta llega por **fincas** (tramos). Para cada finca `f` se conoce exactamente el volumen por calibre `K[f,c]` (preconteo del día anterior).
- Dentro de una finca, los calibres están **mezclados**; el orden de aparición es estocástico.
- Hay `M` salidas (ejemplo: 8). Cada salida tiene un **tipo fijo** (AUTO / BULK / MANUAL) y una velocidad `v_type` (kg/h). La velocidad no depende del calibre.
- Cada salida puede procesar **un solo calibre a la vez**. Se puede asignar más de una salida al mismo calibre.

## Dependencias recomendadas
- Simulación: **SimPy**, `numpy`, `pandas`
- Optimización v1: `pulp` (rápido) o `pyomo` (más extensible)
- Optimización v2 (scheduling con setups): **OR-Tools CP-SAT**

## Entregables esperados del proyecto
1. Simulador con:
   - throughput total y por calibre
   - utilización por salida
   - tiempos ociosos/bloqueo
   - tamaño de cola por calibre (y sizing de buffers)
   - corrida con N semillas para intervalos de confianza
2. Optimizador v1:
   - asignación estática `caliber_by_outlet`
   - métrica robusta `lambda*` y reporte por finca
   - validación en simulación (promedio y percentiles)
3. Optimizador v2:
   - plan de producción por salida con cambios de calibre
   - penalización por setups
   - comparación vs v1 en escenarios con alta variación por finca

## Estructura sugerida de repo
```
lemon_packing/
  README.md
  pyproject.toml
  configs/
    simulator_config.yaml
  src/
    lemon_packing/
      __init__.py
      types.py
      sim/
        simpy_engine.py
        generators.py
        metrics.py
      opt/
        assignment.py
      io/
        loaders.py
        exporters.py
  notebooks/
    01_validate_simulator.ipynb
    02_compare_v1_vs_v2.ipynb
  tests/
    test_smoke_sim.py
```

## Cómo usar estos docs
- Empezar por `01_simulador.md` e implementar el simulador “mínimo viable”.
- Implementar `02_optimizador_v1.md` y enchufarlo a la simulación.
- Recién después avanzar con `03_optimizador_v2.md`.
