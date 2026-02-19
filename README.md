# TesisLo — Simulador de Empaque de Limones

Simulador de eventos discretos (SimPy) para optimización de empaque por calibre.

## Uso

```bash
uv run python main.py                      # usa data/farms.csv
uv run python main.py data/mi_dia.csv      # usa otro CSV
```

## Input: CSV de fincas

El archivo CSV debe tener **4 columnas**:
- `finca`: nombre de la finca
- `80`: kg de calibre 80
- `100`: kg de calibre 100
- `120`: kg de calibre 120

Ejemplo (`data/farms.csv`):
```
finca,80,100,120
F1,5000,8000,3000
F2,3000,12000,5000
F3,4000,6000,8000
```

## Configuración

- `configs/simulator_config.yaml`: parámetros del simulador (turno, buffers, velocidades, asignación)
