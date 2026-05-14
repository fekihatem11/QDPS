PYTHON = python3

## compare — Run QDPS and compare against SETS/DeepGD/RS baselines
compare:
	$(PYTHON) compare_results.py

## run — Full experiment (30 runs per configuration)
run:
	$(PYTHON) run_experiment.py 30

## quick — Quick experiment (5 runs per configuration)
quick:
	$(PYTHON) run_experiment.py 5

## single — Single run (1 run per configuration)
single:
	$(PYTHON) run_experiment.py 1

## subject — Run QDPS on a single subject (usage: make subject S=mnist_LeNet1 N=30 K=100,300,500)
subject:
	$(PYTHON) run_single_subject.py $(S) $(or $(N),30) $(if $(K),$(K))

## table — Generate LaTeX comparison table (QDPS vs SETS)
table:
	$(PYTHON) generate_table.py

## sets — Run SETS baseline on all subjects (30 runs), save to sets_results/
sets:
	$(PYTHON) run_sets_baseline.py

## stat — Wilcoxon Signed-Rank Test: QDPS vs SETS/DeepGD/RS (30 runs)
stat:
	$(PYTHON) statistical_test.py

## help — Show available commands
help:
	@echo ""
	@echo "Available commands:"
	@echo "  make compare  — Compare QDPS against baselines (single run)"
	@echo "  make run      — Full experiment (30 runs)"
	@echo "  make quick    — Quick experiment (5 runs)"
	@echo "  make single   — Single run per configuration"
	@echo "  make subject S=mnist_LeNet1 N=30 K=100,300,500"
	@echo "                — Run QDPS on one subject (N runs, budgets K)"
	@echo "  make table    — Generate LaTeX table (QDPS vs SETS)"
	@echo "  make sets     — Run SETS baseline (30 runs, saves to sets_results/)"
	@echo "  make stat     — Wilcoxon Signed-Rank Test (30 runs vs baselines)"
	@echo "  make clean    — Remove all experiment results"
	@echo ""

## clean — Remove experiment results
clean:
	rm -rf results/

.PHONY: compare run quick single stat help clean
