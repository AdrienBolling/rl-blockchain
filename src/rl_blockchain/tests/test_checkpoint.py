"""Checkpoint save/restore behaviour.

Pins the four defects that were fixed:

1. orbax saves asynchronously -- without `wait_until_finished()` the last save is
   left as an uncommitted `<step>.orbax-checkpoint-tmp` and silently lost.
2. `max_to_keep` alone makes older epochs unreachable; `keep_period` pins them.
3. `opt_state` and `rng_key` are in the checkpoint and must be restored (a resume),
   unless `warm_start` is asked for explicitly.
4. the architecture must come back from `run_config.json`, and a mismatch must
   fail loudly at restore time rather than later inside flax.
"""

import pathlib
import subprocess
import sys
import textwrap
from argparse import Namespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from rl_blockchain.algo.ppo import (create_checkpoint_manager, create_ppo_state, init_ppo_state,
                                    latest_checkpoint_step, load_ppo_state, make_optimizer)
from rl_blockchain.utils.run_config import CONFIG_FILENAME, apply_model_config, save_run_config

N_NODES, LR = 7, 3e-4


def _make(n_nodes=N_NODES, seed=0):
    from rl_blockchain.scripts.env_factory import GenericEnvFactory
    from rl_blockchain.scripts.parser import UpdateValStrat, UpdateDistStrat

    cfg = {"n_nodes": n_nodes, "gat_arch": [8, 8, 8], "voting_nodes": 0,
           "reward_weights": [0.5, 0.5],
           "next_val_type": UpdateValStrat.NO_UPDATE,
           "next_edge_type": UpdateDistStrat.NO_UPDATE}
    model, env, _, _, _ = GenericEnvFactory.create("blockenv", jax.random.PRNGKey(seed), cfg)
    return model, env


def _trained_state(model, env):
    """A state whose optimizer moments are non-zero, so restoring them is observable."""
    state = init_ppo_state(env, seed=0, lr=LR, model=model)
    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, state.params)
    updates, opt_state = make_optimizer(LR).update(grads, state.opt_state)
    return state.replace(params=optax.apply_updates(state.params, updates), opt_state=opt_state)


def _save(directory, state, steps, **kwargs):
    mgr = create_checkpoint_manager(directory, **kwargs)
    for s in steps:
        mgr.save(step=s, args=ocp.args.StandardSave(state))
    mgr.wait_until_finished()
    mgr.close()


# ---------------------------------------------------------------- 1. async save

def test_exiting_after_wait_commits_the_last_checkpoint(tmp_path):
    """Regression: a process that saves then exits must not leave a *.tmp dir."""
    prog = textwrap.dedent(f"""
        import jax, orbax.checkpoint as ocp
        from rl_blockchain.algo.ppo import create_checkpoint_manager, init_ppo_state
        from rl_blockchain.scripts.env_factory import GenericEnvFactory
        from rl_blockchain.scripts.parser import UpdateValStrat, UpdateDistStrat
        cfg = dict(n_nodes={N_NODES}, gat_arch=[8, 8, 8], voting_nodes=0, reward_weights=[.5, .5],
                   next_val_type=UpdateValStrat.NO_UPDATE, next_edge_type=UpdateDistStrat.NO_UPDATE)
        model, env, _, _, _ = GenericEnvFactory.create("blockenv", jax.random.PRNGKey(0), cfg)
        st = init_ppo_state(env, 0, {LR}, model)
        mgr = create_checkpoint_manager(r"{tmp_path / 'run'}", max_to_keep=1)
        try:
            for step in range(3):
                mgr.save(step=step, args=ocp.args.StandardSave(st))
        finally:
            mgr.wait_until_finished()   # the fix
            mgr.close()
    """)
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                         env={**__import__("os").environ, "JAX_PLATFORMS": "cpu"})
    assert out.returncode == 0, out.stderr[-2000:]

    on_disk = sorted(p.name for p in (tmp_path / "run").iterdir())
    assert not any(n.endswith(".orbax-checkpoint-tmp") for n in on_disk), on_disk
    assert latest_checkpoint_step(tmp_path / "run") == 2, on_disk


def test_uncommitted_checkpoint_is_reported_not_ignored(tmp_path):
    """An empty run dir must raise, and mention any lost *.tmp saves."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "7.orbax-checkpoint-tmp").mkdir()
    model, env = _make()
    target = init_ppo_state(env, 0, LR, model)
    with pytest.raises(ValueError, match=r"No committed checkpoints.*orbax-checkpoint-tmp"):
        load_ppo_state(run, jax.random.PRNGKey(0), target=target)


# ------------------------------------------------------- 2. keeping old models

def test_keep_period_retains_old_steps(tmp_path):
    model, env = _make()
    _save(tmp_path / "run", init_ppo_state(env, 0, LR, model), range(6),
          max_to_keep=1, keep_period=2)
    mgr = create_checkpoint_manager(tmp_path / "run", create=False)
    steps = sorted(mgr.all_steps())
    mgr.close()
    assert 5 in steps, f"latest must survive: {steps}"
    assert {0, 2, 4} <= set(steps), f"every 2nd step must be pinned: {steps}"


def test_load_a_specific_earlier_step(tmp_path):
    model, env = _make()
    run = tmp_path / "run"
    mgr = create_checkpoint_manager(run, max_to_keep=5)
    states = {}
    state = init_ppo_state(env, 0, LR, model)
    for step in range(3):
        states[step] = state
        mgr.save(step=step, args=ocp.args.StandardSave(state))
        state = _trained_state(model, env) if step == 0 else state
    mgr.wait_until_finished()
    mgr.close()

    target = init_ppo_state(env, 0, LR, model)
    loaded = load_ppo_state(run, jax.random.PRNGKey(0), target=target, step=1)
    assert jax.tree.all(jax.tree.map(lambda a, b: bool(np.array_equal(a, b)),
                                     loaded.params, states[1].params))


def test_unknown_step_raises_and_lists_available(tmp_path):
    model, env = _make()
    _save(tmp_path / "run", init_ppo_state(env, 0, LR, model), [0, 1], max_to_keep=5)
    target = init_ppo_state(env, 0, LR, model)
    with pytest.raises(ValueError, match=r"Step 42 not found.*Available: \[0, 1\]"):
        load_ppo_state(tmp_path / "run", jax.random.PRNGKey(0), target=target, step=42)


# ----------------------------------------- 3. resume vs warm start

def test_resume_restores_params_optimizer_and_rng(tmp_path):
    model, env = _make()
    saved = _trained_state(model, env)
    _save(tmp_path / "run", saved, [0])

    restored = create_ppo_state(resume_dir=tmp_path / "run", env=env, seed=0, lr=LR, model=model)

    assert jax.tree.all(jax.tree.map(lambda a, b: bool(np.array_equal(a, b)),
                                     restored.params, saved.params)), "weights"
    assert jax.tree.all(jax.tree.map(lambda a, b: bool(np.array_equal(a, b)),
                                     restored.opt_state, saved.opt_state)), "optimizer moments"
    np.testing.assert_array_equal(restored.rng_key, saved.rng_key)
    # optax must accept it: i.e. it is a real ScaleByAdamState, not a raw dict
    make_optimizer(LR).update(restored.params, restored.opt_state)


def test_warm_start_keeps_weights_but_resets_optimizer_and_rng(tmp_path):
    model, env = _make()
    saved = _trained_state(model, env)
    _save(tmp_path / "run", saved, [0])
    fresh = init_ppo_state(env, 0, LR, model)

    warm = create_ppo_state(resume_dir=tmp_path / "run", env=env, seed=0, lr=LR,
                            model=model, warm_start=True)

    assert jax.tree.all(jax.tree.map(lambda a, b: bool(np.array_equal(a, b)),
                                     warm.params, saved.params)), "weights kept"
    assert jax.tree.all(jax.tree.map(lambda a, b: bool(np.array_equal(a, b)),
                                     warm.opt_state, fresh.opt_state)), "moments reset"
    # the saved optimizer state was non-trivial, so 'reset' is a real difference
    assert not jax.tree.all(jax.tree.map(lambda a, b: bool(np.array_equal(a, b)),
                                         saved.opt_state, fresh.opt_state))


# ---------------------------------------- 4. architecture / run config

def test_reading_a_missing_directory_raises_and_creates_nothing(tmp_path):
    missing = tmp_path / "typo"
    with pytest.raises(FileNotFoundError):
        create_checkpoint_manager(missing, create=False)
    assert not missing.exists(), "a mistyped path must not be created"


def test_architecture_mismatch_fails_at_restore(tmp_path):
    model, env = _make()
    _save(tmp_path / "run", init_ppo_state(env, 0, LR, model), [0])

    from rl_blockchain.scripts.env_factory import GenericEnvFactory
    from rl_blockchain.scripts.parser import UpdateValStrat, UpdateDistStrat
    cfg = {"n_nodes": N_NODES, "gat_arch": [32, 32, 32], "voting_nodes": 0,
           "reward_weights": [0.5, 0.5], "next_val_type": UpdateValStrat.NO_UPDATE,
           "next_edge_type": UpdateDistStrat.NO_UPDATE}
    other, env2, _, _, _ = GenericEnvFactory.create("blockenv", jax.random.PRNGKey(0), cfg)

    with pytest.raises(ValueError, match="not compatible with the stored shape"):
        create_ppo_state(resume_dir=tmp_path / "run", env=env2, seed=0, lr=LR, model=other)


def test_run_config_roundtrip(tmp_path):
    from rl_blockchain.scripts.parser import UpdateValStrat, UpdateDistStrat

    args = Namespace(env="blockenv", n_nodes=25, gat_arch=[64, 32, 16], voting_nodes=7,
                     reward_weights=[0.1, 0.9], update_params=UpdateValStrat.ORN_UHL_UPDATE,
                     next_edge_type=UpdateDistStrat.NO_UPDATE,
                     ref_map_file=pathlib.Path("/some/grid.csv"), seed=3, num_epochs=1000)
    path = save_run_config(tmp_path, args)
    assert path.name == CONFIG_FILENAME

    # eval is invoked with the *default* flags; they must be overwritten
    evaluated = Namespace(env="cartpole", n_nodes=200, gat_arch=[1, 1, 1], voting_nodes=0,
                          reward_weights=[1.0, 0.0], update_params=UpdateValStrat.NO_UPDATE,
                          next_edge_type=UpdateDistStrat.ORN_UHL_UPDATE, ref_map_file=None)
    assert apply_model_config(evaluated, tmp_path) is True
    assert evaluated.n_nodes == 25 and evaluated.gat_arch == [64, 32, 16]
    assert evaluated.env == "blockenv" and evaluated.voting_nodes == 7
    assert evaluated.update_params is UpdateValStrat.ORN_UHL_UPDATE
    assert evaluated.next_edge_type is UpdateDistStrat.NO_UPDATE
    assert evaluated.ref_map_file == pathlib.Path("/some/grid.csv")
    # non-model args are informational and must not be copied over
    assert not hasattr(evaluated, "num_epochs")


def test_apply_model_config_is_a_noop_for_older_runs(tmp_path):
    args = Namespace(n_nodes=25)
    assert apply_model_config(args, tmp_path) is False
    assert args.n_nodes == 25
