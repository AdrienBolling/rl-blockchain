#!/bin/bash
#
# Evaluate every trained run under a checkpoint root and merge the results into a
# single comparison table. Each run's env/model config (n_nodes, horizon, gat_arch,
# voting_nodes, reward_weights, update_params, next_edge_type, gini_reward_mode) is
# restored from its run_config.json, so we only pass eval-time knobs here.
#
# Usage:
#   ./eval_all.sh [CKPT_ROOT] [OUT_DIR] [EVAL_EPISODES] [SEED]
# Defaults:
#   CKPT_ROOT=checkpoints/guilain-leduc_rl_blockchain  OUT_DIR=eval_results
#   EVAL_EPISODES=100  SEED=0
#
set -u
export LC_NUMERIC=C

CKPT_ROOT="${1:-checkpoints/guilain-leduc_rl_blockchain}"
OUT_DIR="${2:-eval_results}"
EVAL_EPISODES="${3:-100}"
SEED="${4:-0}"

mkdir -p "$OUT_DIR"

# A run directory is any dir that holds a run_config.json (its orbax step subdirs
# live beneath it). -printf '%h' yields that parent dir.
mapfile -t RUN_DIRS < <(find "$CKPT_ROOT" -type f -name run_config.json -printf '%h\n' | sort -u)

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
  echo "No run_config.json found under '$CKPT_ROOT'." >&2
  echo "Pass the checkpoint root as the first argument, e.g.:" >&2
  echo "  ./eval_all.sh checkpoints/<entity>_<project>" >&2
  exit 1
fi

echo "Found ${#RUN_DIRS[@]} run(s) under $CKPT_ROOT"
echo "Writing per-run results to $OUT_DIR/  (eval_episodes=$EVAL_EPISODES, seed=$SEED)"

for dir in "${RUN_DIRS[@]}"; do
  name=$(basename "$dir")
  out="$OUT_DIR/eval_${name}.json"   # the .csv is written next to it automatically
  echo ">>> eval  $name"
  uv run src/rl_blockchain/scripts/run.py \
      --seed "$SEED" \
      ppo --eval-episodes "$EVAL_EPISODES" \
      eval "$dir" --output "$out" \
    || echo "  !! eval failed for $name (skipping)" >&2
done

# Merge every one-row CSV into a single table + print a compact pivot on the
# mode-INDEPENDENT metrics (the only ones comparable across gini_reward_mode).
uv run python - "$OUT_DIR" <<'PY'
import sys, pathlib
import pandas as pd

out_dir = pathlib.Path(sys.argv[1])
files = sorted(out_dir.glob("eval_*.csv"))
if not files:
    print(f"No eval_*.csv found in {out_dir}", file=sys.stderr)
    raise SystemExit(1)

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
merged = out_dir / "comparison.csv"
df.to_csv(merged, index=False)
print(f"\nMerged {len(files)} run(s) -> {merged}\n")

# Cell-identifying dimensions, in sort/group order (only those actually present).
dims = [c for c in ("gini_reward_mode", "horizon", "gini_lambda") if c in df.columns]
metrics = [c for c in ("gini", "distance", "weighted_original_reward") if c in df.columns]

if dims and metrics:
    view = df[[*dims, *metrics]].sort_values(dims).reset_index(drop=True)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("Comparison (lower gini & lower distance = better):\n")
        print(view.to_string(index=False))

    # One pivot per metric: rows = (mode, horizon), columns = gini_lambda.
    row_dims = [c for c in ("gini_reward_mode", "horizon") if c in df.columns]
    if "gini_lambda" in df.columns and row_dims:
        for m in metrics:
            print(f"\nPivot: {m}  ({' x '.join(row_dims)})  x  gini_lambda\n")
            print(df.pivot_table(index=row_dims, columns="gini_lambda", values=m).to_string())

    if df.get("horizon") is not None and df["horizon"].nunique() > 1:
        print("\nNOTE: gini is worst-normalized, but its *meaning* changes with horizon "
              "(the fairness-window length), so gini is NOT directly comparable across "
              "horizons -- choose horizon by the fairness timescale you want, then compare "
              "gini_lambda *within* a horizon. distance / weighted_original_reward are "
              "comparable across horizons.")
PY
