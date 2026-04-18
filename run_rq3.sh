#!/bin/bash
set -e
cd /home/spotai_small/fzhang/code/liars_dice_tom
eval "$(/home/spotai_small/fzhang/bin/micromamba shell hook --shell bash)"
micromamba activate fzhang-ld

PPO_PATH="checkpoints/ppo_stage2/final_model.zip"
HE2016_DIR="checkpoints/he2016/"
EXPERTS_DIR="checkpoints/he2016_experts/"
OUTPUT_DIR="results/rq3/"
N_GAMES=5000

mkdir -p "$OUTPUT_DIR"
echo "=== RQ3 started: $(date)"

PYTHONNOUSERSITE=1 python -u experiments/rq3_ch_comparison.py     --n_games $N_GAMES     --ppo_path "$PPO_PATH"     --he2016_dir "$HE2016_DIR"     --experts_dir "$EXPERTS_DIR"     --output_dir "$OUTPUT_DIR"     --env_n_dice 5

PYTHONNOUSERSITE=1 python -u analysis/rq3_plots.py     --results_dir "$OUTPUT_DIR"     --plots_dir "$OUTPUT_DIR/plots/"

echo "Done: $(date)"
