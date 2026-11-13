PYTHON = python3

## compare — Run QDPS and compare against SETS/DeepGD/RS baselines
compare:
	cd .. && $(PYTHON) -m qdps.analysis.compare_results

## run — Full experiment (30 runs per configuration)
run:
	cd .. && $(PYTHON) -m qdps.experiments.run_experiment 30

## quick — Quick experiment (5 runs per configuration)
quick:
	cd .. && $(PYTHON) -m qdps.experiments.run_experiment 5

## single — Single run (1 run per configuration)
single:
	cd .. && $(PYTHON) -m qdps.experiments.run_experiment 1

## subject — Run QDPS on a single subject (usage: make subject S=mnist_LeNet1 N=30 K=100,300,500)
subject:
	cd .. && $(PYTHON) -m qdps.experiments.run_single_subject $(S) $(or $(N),30) $(if $(K),$(K))

## table — Generate LaTeX comparison table (QDPS vs SETS)
table:
	cd .. && $(PYTHON) -m qdps.analysis.generate_fdr_table

## sets — Run SETS baseline on all subjects (30 runs), save to sets_results/
sets:
	cd .. && $(PYTHON) -m qdps.experiments.run_sets_baseline

## stat — Wilcoxon Signed-Rank Test: QDPS vs SETS/DeepGD/RS (30 runs)
stat:
	cd .. && $(PYTHON) -m qdps.analysis.statistical_test

## retrain — RQ4 retraining experiment QDPS vs SETS (usage: make retrain S=mnist_LeNet1 K=500 N=5)
retrain:
	cd .. && $(PYTHON) -m qdps.retrain.run_retrain $(or $(S),all-mnist) $(or $(K),500) $(or $(N),5)

## retrain-mnist — Phase-1 validation slice: MNIST LeNet1/5, k=500, QDPS vs SETS, 5 runs
retrain-mnist:
	cd .. && $(PYTHON) -m qdps.retrain.run_retrain all-mnist 500 5

## retrain-table — Build QDPS-vs-SETS comparison LaTeX table from retrain_results/ (usage: make retrain-table K=500)
retrain-table:
	cd .. && $(PYTHON) -m qdps.analysis.generate_retrain_table $(or $(K),500)

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
	@echo "  make retrain S=mnist_LeNet1 K=500 N=5"
	@echo "                — RQ4 retraining experiment (QDPS vs SETS)"
	@echo "  make retrain-mnist — Phase-1 validation slice (MNIST, k=500, 5 runs)"
	@echo "  make retrain-table K=500"
	@echo "                — Build QDPS-vs-SETS comparison LaTeX table"
	@echo "  make clean    — Remove all experiment results"
	@echo ""

## clean — Remove experiment results
clean:
	rm -rf results/

.PHONY: compare run quick single subject table sets stat retrain retrain-mnist retrain-table help clean
