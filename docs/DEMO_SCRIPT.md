# 2–3 Minute Contest Demo Script

1. **Default 48-rack case** — show that the application detects rack count, pod count, IT load and liquid load automatically.
2. **Phase 1** — edit one rack power or HCR and show the load map / pod totals update; approve the phase.
3. **Phase 2** — compare TCS/CDU topology candidates and show candidate CDU loading; record the engineer's choice.
4. **Phase 3** — select quantitatively defined coolant sensitivity candidates; explain that OEM/material approval remains a validation gate.
5. **Phase 4** — show deterministic rack/pod flow, calculated pressure drop and pump power; download the calculation CSV.
6. **Phase 5** — show the transparent sensitivity ranking and decision history; download the design-review report.
7. **Scalability proof** — upload `data/sample_64_racks.csv`; point out that the rack count and pod results are recalculated automatically without code changes.

## One-sentence pitch

> H-LiquidOpt does not ask an LLM to design a cooling system; it structures engineering inputs, pauses for engineer approval at each phase, and sends validated inputs to deterministic heat/hydraulic calculations before comparing design candidates.
