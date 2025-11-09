#!/bin/bash
# Script to run all experiments in parallel on different GPUs

echo "================================================================================"
echo "Running all 4 experiments in parallel on separate GPUs"
echo "================================================================================"

# Run each experiment in the background
echo "Starting exp_train_on_D.py on GPU 0..."
python exp_train_on_D.py > logs_exp_train_on_D.txt 2>&1 &
PID1=$!

echo "Starting exp_train_on_D_m.py on GPU 1..."
python exp_train_on_D_m.py > logs_exp_train_on_D_m.txt 2>&1 &
PID2=$!

echo "Starting exp_plot_A_heatmap.py on GPU 2..."
python exp_plot_A_heatmap.py > logs_exp_plot_A_heatmap.txt 2>&1 &
PID3=$!

echo "Starting exp_vanish_grad.py on GPU 3..."
python exp_vanish_grad.py > logs_exp_vanish_grad.txt 2>&1 &
PID4=$!

echo ""
echo "All experiments started!"
echo "  exp_train_on_D.py       (PID: $PID1) -> logs_exp_train_on_D.txt"
echo "  exp_train_on_D_m.py     (PID: $PID2) -> logs_exp_train_on_D_m.txt"
echo "  exp_plot_A_heatmap.py   (PID: $PID3) -> logs_exp_plot_A_heatmap.txt"
echo "  exp_vanish_grad.py      (PID: $PID4) -> logs_exp_vanish_grad.txt"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs_*.txt"
echo ""
echo "Wait for all to complete with:"
echo "  wait $PID1 $PID2 $PID3 $PID4"
echo "================================================================================"

# Wait for all background jobs to complete
wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "================================================================================"
echo "All experiments completed!"
echo "================================================================================"
echo "Results saved in results/ directory:"
echo "  - results/train_on_D/"
echo "  - results/exp_train_on_D_m/"
echo "  - results/plot_A_heatmap/"
echo "  - results/vanish_grad/"
echo "================================================================================"
