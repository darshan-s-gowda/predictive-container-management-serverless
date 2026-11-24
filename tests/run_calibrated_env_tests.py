"""Quick inline test runner for ppo_calibrated_env (no pytest needed)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, 'src')

ROOT = Path('.')
DATA = ROOT / 'data' / 'azure_serverless_workload.csv'

from ppo_calibrated_env import (
    make_training_env, CalibratedServerlessEnv,
    build_persistence_forecast, ACTION_MAP, OBS_DIM, N_ACTIONS
)
from serverless_simulator import ServerlessSimulator

results = []

def chk(name, ok):
    results.append((name, bool(ok)))

env = make_training_env(data_csv=DATA, episode_len=288, seed=42)

# 1 obs shape
obs, _ = env.reset(seed=0)
chk('obs_shape_correct', obs.shape == (OBS_DIM,) and np.all(np.isfinite(obs)))

# 2 action space
chk('action_space_n_eq_6', env.action_space.n == 6)

# 3 step finite
env.reset(seed=0)
obs, r, _, _, _ = env.step(0)
chk('step_obs_reward_finite', np.all(np.isfinite(obs)) and np.isfinite(r))

# 4 no test leakage
df = pd.read_csv(DATA)
n = len(df)
split = int(n * 0.85)
chk('no_test_data_in_episodes', all(len(ep[0]) <= split for ep in env.episodes))

# 5 action map = simulator actions
chk('action_map_matches_simulator',
    set(ACTION_MAP.values()) == set(ServerlessSimulator.ACTION_DELTAS.keys()))

# 6 all 6 actions produce finite output
env2 = make_training_env(data_csv=DATA, episode_len=50, seed=42)
all_ok = True
for a in range(6):
    env2.reset(seed=a)
    obs2, r2, _, _, _ = env2.step(a)
    if not (np.all(np.isfinite(obs2)) and np.isfinite(r2)):
        all_ok = False
chk('all_6_actions_finite', all_ok)

# 7 heavy overload => lower reward than light
light = np.full(50, 1.0, dtype=np.float32)
heavy = np.full(50, 500.0, dtype=np.float32)
el = CalibratedServerlessEnv([(light, build_persistence_forecast(light))], seed=0)
eh = CalibratedServerlessEnv([(heavy, build_persistence_forecast(heavy))], seed=0)

def run_ep(e):
    obs, _ = e.reset(seed=0)
    tot = 0.0
    for _ in range(49):
        obs, r, done, _, _ = e.step(0)
        tot += r
        if done:
            break
    return tot

chk('heavy_overload_worse_reward', run_ep(eh) < run_ep(el))

# 8 determinism under same seed
e1 = make_training_env(data_csv=DATA, episode_len=100, seed=99)
e2 = make_training_env(data_csv=DATA, episode_len=100, seed=99)
o1, _ = e1.reset(seed=0)
o2, _ = e2.reset(seed=0)
det = np.array_equal(o1, o2)
for a in range(10):
    ob1, r1, _, _, _ = e1.step(a % 6)
    ob2, r2, _, _, _ = e2.step(a % 6)
    if not (np.array_equal(ob1, ob2) and r1 == r2):
        det = False
chk('determinism_same_seed', det)

# 9 invalid action raises
env.reset(seed=0)
raised = False
try:
    env.step(99)
except (ValueError, KeyError):
    raised = True
chk('invalid_action_raises', raised)

# --- print results ---
print()
print('=' * 55)
passed = sum(1 for _, v in results if v)
print(f'RESULTS: {passed}/{len(results)} PASS')
print('=' * 55)
for name, ok in results:
    status = 'OK  ' if ok else 'FAIL'
    print(f'  [{status}] {name}')

if passed < len(results):
    sys.exit(1)
