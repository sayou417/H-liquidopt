# Validation Notes

## Inputs that are intentionally synthetic in the demo

- 48-rack default layout and support-rack power
- preliminary pipe diameters and route lengths
- lumped minor-loss K values
- rack internal ΔP reference of 120 kPa
- pump/motor efficiencies
- TCS supply/return temperatures

These are replaceable project inputs, not claimed OEM values.

## Inputs that must be validated before engineering use

1. IT rack design power and liquid heat-capture ratio
2. OEM minimum/maximum flow and pressure-flow curve
3. coolant formulation approval and temperature-dependent properties
4. wetted-material and seal compatibility
5. actual pipe routing, fittings and equipment losses
6. CDU performance map and control envelope
7. project redundancy and operating criteria

## Rack-count independence

The calculation engine does not contain a 48-rack constant. The default CSV contains 48 rows for the contest demonstration only. Uploaded datasets can contain arbitrary rack counts and arbitrary pod names. Unit tests exercise 12, 24, 48, 64 and 96 rack datasets.

## Heat-map interpretation

The layout visualization is a **design heat-load density map**, not a CFD temperature map. If `row` and `col` are absent, thermal and hydraulic totals still calculate, while the spatial heat map is omitted.
