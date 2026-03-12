# AGENTS.md

## Project at a glance
- This repo trains PPO agents (JAX/Flax) for a custom blockchain validator-selection environment, with optional `CartPole` fallback.
- Main runtime entrypoint is `src/rl_blockchain/scripts/run.py`, which dispatches `ppo train` vs `ppo eval`.
- Core stack is: env construction (`scripts/env_factory.py`) -> rollout/train/eval (`algo/ppo.py`) -> model (`model.py`) -> checkpoint/logging (`orbax`, `wandb`, `logs/`).

## Key architecture and data flow
- `scripts/parser.py` defines the full CLI contract; many defaults are operationally important (paths, wandb naming, training hyperparameters).
- `scripts/ppo_func.py` is the orchestration layer:
  - Builds env/model via `get_env_config(...)` + `GenericEnvFactory.create(...)`.
  - Builds per-epoch schedules from 1- or 2-value CLI lists (`--learning-rate`, `--clip-ratio`, `--entropy-coef`).
  - Creates checkpoint dir as `checkpoints/<entity>_<project>/run_<N>` using wandb run count.
- `algo/ppo.py` contains the real PPO implementation:
  - Uses subset actions (`sample_subset_with_logp`, `logp_prefix_pl`) for selecting `k` validators, not a single discrete action.
  - Uses `train_epoch(...)` for vmapped rollout + minibatch PPO updates.
  - Uses Orbax `CheckpointManager`; `load_ppo_state(...)` restores `params` and re-seeds RNG.

## Environment-specific patterns
- `BlockEnv/BlockEnv.py` expects boolean mask actions with exactly `state.nb_val` selected nodes; illegal masks end episode and return `null_reward()`.
- Observations are `jraph.GraphsTuple` with node features from stake-history normalization (`state_params.get_stake_distribution`).
- Reward is weighted composite of Gini fairness + distance (`BlockEnv/rewards.py`) using `EnvParams.rewards_weights`.
- Validator-count dynamics are configured by `--update-params` (`NO_UPDATE`, `THRESHOLD_UPDATE`, `ORN_UHL_UPDATE`) in `scripts/env_factory.py`.

## Developer workflows (actual repo usage)
- Install/sync deps with `uv` (see `pyproject.toml`, `uv.lock`).
- Default training command is in `default_train.sh`:
  - `uv run ./src/rl_blockchain/scripts/run.py ppo train`
- Typical eval command:
  - `uv run ./src/rl_blockchain/scripts/run.py ppo eval <chkpt_dir>`
- Profiling helper entrypoint:
  - `uv run ./src/rl_blockchain/tests/run_test.py profiling blockEnv --n-steps 100000 --n-envs 10`

## Conventions that matter when editing
- Prefer wiring new environment behavior through `GenericEnvFactory` builders rather than branching in training code.
- Keep JAX functions pure/jittable; many functions are `@jax.jit` and rely on shape/static assumptions.
- Logging is dual-channel: Python `logging` files under `logs/` + structured `wandb.log(...)` inside training/eval loops.
- Generated artifacts (`wandb/`, `logs/`, `checkpoints/`) are outputs; do not treat them as source of truth.
- File naming has legacy quirks (e.g., `NormailzationWrapper.py` typo) that are referenced by imports; rename carefully.

## Useful files to read first
- `src/rl_blockchain/scripts/run.py`
- `src/rl_blockchain/scripts/parser.py`
- `src/rl_blockchain/scripts/ppo_func.py`
- `src/rl_blockchain/algo/ppo.py`
- `src/rl_blockchain/scripts/env_factory.py`
- `src/rl_blockchain/BlockEnv/BlockEnv.py`
- `src/rl_blockchain/BlockEnv/state_params.py`
- `src/rl_blockchain/BlockEnv/rewards.py`

