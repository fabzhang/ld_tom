#!/bin/bash
# run_rq1_pipeline.sh — Full RQ1 pipeline, end-to-end.
#
# Stages:
#   1. (Already running) Expert training: 4x PPO specialists in screen he2016_experts
#   2. Wait for experts to finish, then train inference net
#   3. Run full 5-agent RQ1 tournament (E1a + E1b + E1c)
#   4. Generate plots
#
# Run via: screen -dmS rq1_pipeline bash run_rq1_pipeline.sh

set -e
cd /home/spotai_small/fzhang/code/liars_dice_tom

PYTHON="PYTHONNOUSERSITE=1 /home/spotai_small/fzhang/bin/micromamba run -n fzhang-ld python"
EXPERTS_DIR="checkpoints/he2016_experts"
HE2016_DIR="checkpoints/he2016"
PPO_PATH="checkpoints/ppo_stage2/final_model.zip"
RESULTS_DIR="results/rq1_final"
LOG_DIR="logs"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

echo "============================================================"
echo "RQ1 Pipeline started: $(date)"
echo "============================================================"

# ---------------------------------------------------------------
# Stage 2: Wait for all 4 experts to finish
# ---------------------------------------------------------------
echo "[pipeline] Waiting for expert training to complete..."
EXPERTS_NEEDED="expert_random.zip expert_heuristic.zip expert_bayesian.zip expert_ppo.zip"
while true; do
    all_done=true
    for f in $EXPERTS_NEEDED; do
        if [ ! -f "$EXPERTS_DIR/$f" ]; then
            all_done=false
            break
        fi
    done
    if $all_done; then
        echo "[pipeline] All experts found: $(date)"
        break
    fi
    # Show which experts are done so far
    done_count=$(ls "$EXPERTS_DIR"/expert_*.zip 2>/dev/null | wc -l)
    echo "[pipeline] $(date): $done_count/4 experts done, waiting..."
    sleep 300  # check every 5 min
done

# ---------------------------------------------------------------
# Stage 3: Train inference net
# ---------------------------------------------------------------
echo ""
echo "[pipeline] Stage 3: Training He2016 inference net... $(date)"
eval $PYTHON training/he2016_training.py \
    --experts_dir "$EXPERTS_DIR" \
    --ppo_path "$PPO_PATH" \
    --output_dir "$HE2016_DIR" \
    --n_games_per_type 2000 \
    --n_epochs 50 \
    --env_n_dice 5 \
    2>&1 | tee "$LOG_DIR/he2016_inference.log"
echo "[pipeline] Inference net done: $(date)"

# ---------------------------------------------------------------
# Stage 4: Full RQ1 tournament (5 agents: random/heuristic/bayesian/ppo/he2016)
# ---------------------------------------------------------------
echo ""
echo "[pipeline] Stage 4: Running full RQ1 tournament (E1a)... $(date)"
eval $PYTHON experiments/rq1_population_dynamics.py \
    --experiment e1a \
    --n_games 1000 \
    --env_n_dice 5 \
    --ppo_path "$PPO_PATH" \
    --he2016_dir "$HE2016_DIR" \
    --experts_dir "$EXPERTS_DIR" \
    --results_dir "$RESULTS_DIR" \
    2>&1 | tee "$LOG_DIR/rq1_e1a.log"

echo ""
echo "[pipeline] Running E1b population sweep... $(date)"
for focal in bayesian ppo he2016; do
    eval $PYTHON experiments/rq1_population_dynamics.py \
        --experiment e1b \
        --focal "$focal" \
        --n_games 500 \
        --env_n_dice 5 \
        --ppo_path "$PPO_PATH" \
        --he2016_dir "$HE2016_DIR" \
        --experts_dir "$EXPERTS_DIR" \
        --results_dir "$RESULTS_DIR" \
        2>&1 | tee -a "$LOG_DIR/rq1_e1b.log"
done

echo ""
echo "[pipeline] Running E1c cyclic dominance analysis... $(date)"
eval $PYTHON experiments/rq1_population_dynamics.py \
    --experiment e1c \
    --env_n_dice 5 \
    --ppo_path "$PPO_PATH" \
    --he2016_dir "$HE2016_DIR" \
    --experts_dir "$EXPERTS_DIR" \
    --results_dir "$RESULTS_DIR" \
    2>&1 | tee "$LOG_DIR/rq1_e1c.log"

# ---------------------------------------------------------------
# Stage 5: Generate plots
# ---------------------------------------------------------------
echo ""
echo "[pipeline] Stage 5: Generating plots... $(date)"
eval $PYTHON analysis/rq1_plots.py \
    --results_dir "$RESULTS_DIR" \
    --plots_dir "$RESULTS_DIR/plots" \
    2>&1 | tee "$LOG_DIR/rq1_plots.log"

echo ""
echo "============================================================"
echo "RQ1 Pipeline COMPLETE: $(date)"
echo "Results: $RESULTS_DIR/"
echo "Plots:   $RESULTS_DIR/plots/"
echo "============================================================"
