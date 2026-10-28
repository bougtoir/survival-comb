# Survival Comb

Information-preserving and reconstructible representation of
time-to-event data. Standalone methodological study focused on
Kaplan–Meier / time-to-event visualization.

- `scripts/comb.py` — encoding spec, renderer, raster decoder,
  comparator renderers.
- `scripts/scenarios.py` — simulation scenario generators (Study 2).
- `scripts/run_simulations.py` — reconstruction benchmark (Study 3).
- `scripts/run_degradation.py` — transmission robustness (Study 4).
- `scripts/metrics.py` — derived survival quantities + recovery report.
- `docs/THEORY.md` — formal framework: graphical non-identifiability,
  reconstructibility levels, invertibility propositions (Study 1).
- `docs/LEVELS.md`, `docs/GATE_DECISION.md` — level spec + gate verdict.
- `tests/` — exact round-trip tests (Phase 3).

Reproduce: `make all` (needs `pip install -r requirements.txt`);
`make test`. In the public code package the `docs` target is skipped
(the manuscript builder is not part of the public scope); all
canonical results, tables and figures regenerate under `FINAL_RESULTS/`,
`tables/` and `figures/`.
