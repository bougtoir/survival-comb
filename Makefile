PY ?= python3

all: final sims degradation levels figs tables docs

# Canonical replication counts (the submitted results were generated
# with these settings; do not lower them).
final:
	$(PY) scripts/run_final.py
	REA_REPS=10 $(PY) scripts/run_reanalysis.py
	$(PY) scripts/run_comparator.py

sims:
	$(PY) scripts/run_simulations.py

degradation:
	DEG_REPS=50 $(PY) scripts/run_degradation.py

levels:
	LEVEL_REPS=10 $(PY) scripts/run_levels.py

figs: final sims degradation levels
	$(PY) scripts/make_figures.py

tables: final sims degradation levels
	$(PY) scripts/make_tables.py

docs: figs tables
	@if [ -f scripts/build_docx.py ]; then \
		$(PY) scripts/build_docx.py; \
	else \
		echo "docs target skipped: build_docx.py is not in the public code package"; \
	fi

test:
	$(PY) -m pytest tests/ -q

.PHONY: all sims degradation levels figs tables docs test
