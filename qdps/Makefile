PYTHON = python3 -P

## compare — Run QDPS and compare against SETS/DeepGD/RS baselines
compare:
	$(PYTHON) -m qdps.analysis.compare_results

## run — Full experiment (30 runs per configuration)
run:
	$(PYTHON) -m qdps.experiments.run_experiment 30

## quick — Quick experiment (5 runs per configuration)
quick:
	$(PYTHON) -m qdps.experiments.run_experiment 5

## single — Single run (1 run per configuration)
single:
	$(PYTHON) -m qdps.experiments.run_experiment 1

## subject — Run QDPS on a single subject (usage: make subject S=mnist_LeNet1 N=30 K=100,300,500)
subject:
	$(PYTHON) -m qdps.experiments.run_single_subject $(S) $(or $(N),30) $(if $(K),$(K))

## table — Generate LaTeX comparison table (QDPS vs SETS)
table:
	$(PYTHON) -m qdps.analysis.generate_fdr_table

## sets — Run SETS baseline on all subjects (30 runs), save to sets_results/
sets:
	$(PYTHON) -m qdps.experiments.run_sets_baseline

## stat — Wilcoxon Signed-Rank Test: QDPS vs SETS/DeepGD/RS (30 runs)
stat:
	$(PYTHON) -m qdps.analysis.statistical_test

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
