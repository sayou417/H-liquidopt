# H-LiquidOpt

**Human-in-the-Loop D2C liquid-cooling preliminary design support prototype**

H-LiquidOpt is a contest PoC that connects rack/OEM design inputs to a staged engineering workflow:

`Rack data → heat-load split → TCS/CDU candidates → coolant sensitivity → deterministic hydraulics → engineer review`

## What changed in the final prototype

- **Rack-count agnostic:** 48 racks are only the demo dataset. The engine accepts any CSV row count (tested at 12, 24, 48, 64 and 96 racks).
- **Dynamic pods:** pod names and counts are read from the uploaded data rather than hard-coded.
- **Dynamic row concurrency:** hydraulic row flow uses the maximum number of liquid-cooled racks in a physical row when `row` is supplied.
- **Input validation:** required columns, duplicate IDs, HCR range, numeric values and duplicated row/col positions are checked before calculation.
- **Human-in-the-Loop gates:** Phase 1–4 approvals and engineer notes are recorded in the session.
- **Transparent placeholders:** rack internal pressure drop remains explicitly synthetic until replaced by an OEM pressure-flow curve.
- **Downloadable outputs:** hydraulic CSV, current rack dataset, and a design-review Markdown report.

## Required rack CSV fields

| Column | Required | Meaning |
|---|---|---|
| `rack_id` | Yes | Unique rack identifier |
| `pod` | Yes | Cooling/service zone |
| `rack_type` | Yes | User-defined label |
| `it_power_kw` | Yes | Rack design IT power |
| `hcr` | Yes | Liquid heat-capture ratio, 0–1 |
| `row` | Recommended | Physical row for heat map / row-flow grouping |
| `col` | Recommended | Physical column for heat map |

The application includes a downloadable CSV template.

## Calculation scope

The deterministic engine currently demonstrates:

- `Q_liquid = P_IT × HCR`
- `Q_air = P_IT × (1-HCR)`
- `m_dot = Q / (Cp × ΔT)`
- Darcy-Weisbach straight-pipe pressure loss
- lumped minor-loss coefficients
- preliminary pump electric power
- CDU load screening
- transparent hydraulic sensitivity ranking

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest -q
```

The final test suite includes multiple rack counts to prove the engine is not tied to 48 racks.

## Important limitation

This is a **preliminary design PoC**, not a construction or procurement tool. Final design requires project-specific OEM curves, coolant approval, material compatibility, water chemistry, freeze protection, actual pipe routing, equipment performance maps and qualified engineering review.
