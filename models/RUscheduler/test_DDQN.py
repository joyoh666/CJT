import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from typing import Tuple, Dict, List
from IPython.display import display

import argparse
import random

from .DDQNscheduler import GRUQNetwork, MLPQNetwork
from .test_environment import EnvConfig, WindowStore, RUWindowEnv


STATE_NAMES = ["RU1", "RU2", "RU3", "RU12", "RU13", "RU23"]
N_ACTIONS = 6

RU_ANTS: Dict[int, List[int]] = {
    1: [0, 1],
    2: [2, 3],
    3: [4, 5],
}

STATE_RUS: Dict[int, List[int]] = {
    0: [1],
    1: [2],
    2: [3],
    3: [1, 2],
    4: [1, 3],
    5: [2, 3],
}

def make_device(use_cpu: bool, gpu_id: int) -> torch.device:
    if (not use_cpu) and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

def load_checkpoint_and_model(ckpt_path: str, device: torch.device) -> Tuple[dict, nn.Module]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "q_net" not in ckpt:
        raise KeyError("Checkpoint does not contain 'q_net'. Expected DQN checkpoint.")

    sd = ckpt["q_net"]
    if any(k.startswith("gru.") for k in sd.keys()):
        meta = ckpt.get("model_meta", {})
        if meta:
            obs_dim = int(meta.get("obs_dim", sd["gru.weight_ih"].shape[1]))
            hidden_size = int(meta.get("hidden_size", sd["gru.weight_hh"].shape[1]))
            n_actions = int(meta.get("n_actions", sd["q_head.weight"].shape[0]))
        else:
            obs_dim = int(sd["gru.weight_ih"].shape[1])
            hidden_size = int(sd["gru.weight_hh"].shape[1])
            n_actions = int(sd["q_head.weight"].shape[0])

        model = GRUQNetwork(obs_dim=obs_dim, hidden_size=hidden_size, n_actions=n_actions).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        print(f"[MODEL] GRU obs_dim={obs_dim} hidden={hidden_size} actions={n_actions}")
        return ckpt, model

    if all(k in sd for k in ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"]):
        obs_dim = int(sd["net.0.weight"].shape[1])
        h1 = int(sd["net.0.weight"].shape[0])
        h2 = int(sd["net.2.weight"].shape[0])
        n_actions = int(sd["net.4.weight"].shape[0])

        model = MLPQNetwork(obs_dim=obs_dim, h1=h1, h2=h2, n_actions=n_actions).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        print(f"[MODEL] MLP obs_dim={obs_dim} h1={h1} h2={h2} actions={n_actions}")
        return ckpt, model

    raise ValueError("Unsupported q_net architecture in checkpoint.")

def decide_action(policy_name: str, env: RUWindowEnv, obs: np.ndarray, model: Optional[nn.Module], h: Optional[torch.Tensor], device: torch.device) -> Tuple[int, Optional[torch.Tensor]]:
    if policy_name == "model_greedy":
        assert model is not None
        obs_t = torch.from_numpy(obs).to(device).view(1, -1)
        with torch.no_grad():
            if isinstance(model, GRUQNetwork):
                assert h is not None
                q, h_next = model(obs_t, h)
                a = int(torch.argmax(q, dim=-1).item())
                return a, h_next
            q = model(obs_t)
            a = int(torch.argmax(q, dim=-1).item())
            return a, h

    if policy_name == "random":
        return int(np.random.randint(0, N_ACTIONS)), h

    if policy_name == "oracle_next_power":
        seg = int(env._seg_k) + 1
        p = env.all_state_power_segment(seg)
        conn_before = int(env._conn_state)
        score = p - float(env.cfg.handover_penalty) * np.array([0 if a == conn_before else 1 for a in range(N_ACTIONS)], dtype=np.float32)
        return int(np.argmax(score)), h

    if policy_name.startswith("fixed_"):
        tag = policy_name.split("_", 1)[1]
        return int(STATE_NAMES.index(tag)), h

    raise ValueError(f"Unknown policy_name={policy_name}")

def run_episode(env: RUWindowEnv, ep_idx: int, policy_name: str, model: Optional[nn.Module], device: torch.device, keep_trace: bool = False) -> Dict:
    obs = env.reset(ep_idx=ep_idx)
    h = model.init_hidden(1, device) if (model is not None and policy_name == "model_greedy" and isinstance(model, GRUQNetwork)) else None

    recs: List[Dict] = []
    done = False
    ep_return = 0.0
    ep_best_sum = 0.0
    ep_ho_sum = 0.0
    steps = 0

    while not done:
        k = int(env._seg_k)
        obs_full = env.ru_gain_segment(k)
        next_power = env.all_state_power_segment(k + 1)

        action, h = decide_action(policy_name, env, obs, model, h, device)
        obs2, reward, done, info = env.step(action)

        ep_return += float(reward)
        ep_best_sum += float(info["is_best_action"])
        ep_ho_sum += float(info["handover_attempt"])
        steps += 1

        if keep_trace:
            recs.append(
                {
                    "step": steps - 1,
                    "action": int(action),
                    "reward": float(reward),
                    "is_best_action": int(info["is_best_action"]),
                    "handover_attempt": int(info["handover_attempt"]),
                    "obs_ru1": float(obs[:9].reshape(3, -1)[0, -1]),
                    "obs_ru2": float(obs[:9].reshape(3, -1)[1, -1]),
                    "obs_ru3": float(obs[:9].reshape(3, -1)[2, -1]),
                    "full_ru1": float(obs_full[0]),
                    "full_ru2": float(obs_full[1]),
                    "full_ru3": float(obs_full[2]),
                    "next_p_ru1": float(next_power[0]),
                    "next_p_ru2": float(next_power[1]),
                    "next_p_ru3": float(next_power[2]),
                    "next_p_ru12": float(next_power[3]),
                    "next_p_ru13": float(next_power[4]),
                    "next_p_ru23": float(next_power[5]),
                }
            )
        obs = obs2

    out = {
        "window_index": int(ep_idx),
        "n_steps": int(steps),
        "reward_per_step": float(ep_return / max(steps, 1)),
        "best_action_ratio": float(ep_best_sum / max(steps, 1)),
        "handover_per_step": float(ep_ho_sum / max(steps, 1)),
    }
    if keep_trace:
        out["trace"] = recs
    return out

def evaluate_policy(store: WindowStore, env_cfg: EnvConfig, indices: List[int], policy_name: str, model: Optional[nn.Module], device: torch.device) -> pd.DataFrame:
    env = RUWindowEnv(store=store, cfg=env_cfg)
    rows = []
    for ep_idx in indices:
        rows.append(run_episode(env, int(ep_idx), policy_name, model, device, keep_trace=False))
    return pd.DataFrame(rows)


def summarize_policy(df: pd.DataFrame, policy_name: str, split_name: str) -> Dict:
    return {
        "split": split_name,
        "policy": policy_name,
        "n_episodes": int(len(df)),
        "reward_mean_per_step": float(df["reward_per_step"].mean()),
        "best_action_ratio": float(df["best_action_ratio"].mean()),
        "handover_per_step": float(df["handover_per_step"].mean()),
    }

def plot_trajectory_inline(trace: Dict, store: WindowStore, split_name: str, dataset_name: str, seg_len_ticks: int):
    recs = trace["trace"]
    ep_idx = int(trace["window_index"])

    t = np.arange(len(recs)) * (store.dt * float(seg_len_ticks))
    actions = np.array([r["action"] for r in recs], dtype=int)

    full1 = np.array([r["full_ru1"] for r in recs], dtype=float)
    full2 = np.array([r["full_ru2"] for r in recs], dtype=float)
    full3 = np.array([r["full_ru3"] for r in recs], dtype=float)

    nextp1 = np.array([r["next_p_ru1"] for r in recs], dtype=float)
    nextp2 = np.array([r["next_p_ru2"] for r in recs], dtype=float)
    nextp3 = np.array([r["next_p_ru3"] for r in recs], dtype=float)

    obs1 = np.array([r["obs_ru1"] for r in recs], dtype=float)
    obs2 = np.array([r["obs_ru2"] for r in recs], dtype=float)
    obs3 = np.array([r["obs_ru3"] for r in recs], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, full1, label="Full RU1 gain", lw=1.8)
    axes[0].plot(t, full2, label="Full RU2 gain", lw=1.8)
    axes[0].plot(t, full3, label="Full RU3 gain", lw=1.8)
    axes[0].plot(t, obs1, "--", alpha=0.8, label="Obs RU1 (masked, latest bin)")
    axes[0].plot(t, obs2, "--", alpha=0.8, label="Obs RU2 (masked, latest bin)")
    axes[0].plot(t, obs3, "--", alpha=0.8, label="Obs RU3 (masked, latest bin)")
    axes[0].set_ylabel("Gain")
    axes[0].set_title(f"[{split_name}] window={ep_idx} | dataset={dataset_name}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(t, nextp1, label="Next RU1 power", lw=2)
    axes[1].plot(t, nextp2, label="Next RU2 power", lw=2)
    axes[1].plot(t, nextp3, label="Next RU3 power", lw=2)
    axes[1].set_ylabel("Next-Slot Power")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].step(t, actions, where="post", lw=2, label="Selected action")
    axes[2].set_yticks(np.arange(N_ACTIONS))
    axes[2].set_yticklabels(STATE_NAMES)
    axes[2].set_ylabel("Action")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    plt.show()


def plot_policy_comparison_inline(df_cmp: pd.DataFrame, split_name: str):
    show = df_cmp[df_cmp["split"] == split_name].copy()
    show = show.sort_values("reward_mean_per_step", ascending=False).reset_index(drop=True)
    x = np.arange(len(show))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].bar(x, show["reward_mean_per_step"].to_numpy())
    axes[0].set_ylabel("Reward/step")
    axes[0].set_title(f"Policy Comparison ({split_name})")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, show["best_action_ratio"].to_numpy())
    axes[1].set_ylabel("Best Action Ratio")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, show["handover_per_step"].to_numpy())
    axes[2].set_ylabel("Handover/step")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(show["policy"].tolist(), rotation=20, ha="right")

    fig.tight_layout()
    plt.show()

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Test DQN-smallGRU-easiest model on train/test scenarios.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cpu", action="store_true", default=False)
    ap.add_argument("--gpu-id", type=int, default=3)
    ap.add_argument("--num-scenarios", type=int, default=20)
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print("[ARGS] Ignoring unknown args (likely Jupyter):", unknown)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = make_device(use_cpu=bool(args.cpu), gpu_id=int(args.gpu_id))
    print(f"[DEVICE] {device}")

    ckpt, model = load_checkpoint_and_model(args.ckpt, device)

    model_n_actions = int(getattr(model, "n_actions", -1))
    if model_n_actions != N_ACTIONS:
        raise ValueError(
            f"Checkpoint/action mismatch: model has {model_n_actions} actions but this test code expects {N_ACTIONS} "
            f"(RU1,RU2,RU3,RU12,RU13,RU23). Use a 6-action checkpoint trained with Final_Train_260316."
        )

    store = WindowStore(str(ckpt["window_dir"]))
    env_raw = dict(ckpt.get("env_cfg", {}))
    valid_keys = set(EnvConfig.__dataclass_fields__.keys())
    env_cfg = EnvConfig(**{k: env_raw[k] for k in env_raw.keys() if k in valid_keys})

    expected_obs_dim = 3 * int(env_cfg.obs_hist_bins) + N_ACTIONS
    model_obs_dim = int(getattr(model, "obs_dim", expected_obs_dim))
    if model_obs_dim != expected_obs_dim:
        if model_obs_dim >= N_ACTIONS and ((model_obs_dim - N_ACTIONS) % 3 == 0):
            env_cfg.obs_hist_bins = int((model_obs_dim - N_ACTIONS) // 3)
            expected_obs_dim = 3 * int(env_cfg.obs_hist_bins) + N_ACTIONS
            print(f"[ENV] adjusted obs_hist_bins to {env_cfg.obs_hist_bins} to match checkpoint obs_dim={model_obs_dim}")
        else:
            raise ValueError(
                f"Checkpoint/obs mismatch: model obs_dim={model_obs_dim}, but env obs_dim={expected_obs_dim}."
            )

    test_indices = list(map(int, ckpt.get("test_indices", [])))
    train_indices = list(map(int, ckpt.get("train_indices", [])))
    if len(test_indices) == 0 or len(train_indices) == 0:
        raise ValueError("Checkpoint must contain both train_indices and test_indices.")

    print(f"[DATA] window_dir={ckpt['window_dir']}")
    print(f"[DATA] X shape={store.X.shape} dt={store.dt:.6f}s")
    print(f"[ENV] seg_len_ticks={env_cfg.seg_len_ticks} obs_dim={3 * env_cfg.obs_hist_bins + N_ACTIONS} actions={N_ACTIONS}")

    n_total = max(int(args.num_scenarios), 20)
    n_test = min(len(test_indices), max(1, n_total // 2))
    n_train = min(len(train_indices), max(1, n_total - n_test))
    while (n_test + n_train) < n_total:
        if n_test < len(test_indices):
            n_test += 1
        elif n_train < len(train_indices):
            n_train += 1
        else:
            break

    rng = np.random.RandomState(args.seed)
    pick_test = sorted(rng.choice(np.array(test_indices, dtype=np.int64), size=n_test, replace=False).tolist())
    pick_train = sorted(rng.choice(np.array(train_indices, dtype=np.int64), size=n_train, replace=False).tolist())

    print(f"[SCENARIOS] showing test={len(pick_test)} + train={len(pick_train)} = {len(pick_test) + len(pick_train)}")

    env = RUWindowEnv(store=store, cfg=env_cfg)

    print("\n[DISPLAY] TEST scenarios")
    for ep_idx in pick_test:
        res = run_episode(env, int(ep_idx), "model_greedy", model, device, keep_trace=True)
        dataset_name = str(store.manifest.loc[int(ep_idx), "dataset"]) if "dataset" in store.manifest.columns else "unknown"
        plot_trajectory_inline(res, store, split_name="test", dataset_name=dataset_name, seg_len_ticks=int(env_cfg.seg_len_ticks))

    print("\n[DISPLAY] TRAIN scenarios")
    for ep_idx in pick_train:
        res = run_episode(env, int(ep_idx), "model_greedy", model, device, keep_trace=True)
        dataset_name = str(store.manifest.loc[int(ep_idx), "dataset"]) if "dataset" in store.manifest.columns else "unknown"
        plot_trajectory_inline(res, store, split_name="train", dataset_name=dataset_name, seg_len_ticks=int(env_cfg.seg_len_ticks))

    baselines = ["model_greedy", "random", "fixed_RU1", "fixed_RU2", "fixed_RU3", "fixed_RU12", "fixed_RU13", "fixed_RU23", "oracle_next_power"]
    rows = []
    for split_name, idxs in [("test", test_indices), ("train", train_indices)]:
        for p in baselines:
            mdl = model if p == "model_greedy" else None
            df = evaluate_policy(store, env_cfg, idxs, p, mdl, device)
            rows.append(summarize_policy(df, p, split_name))

    df_cmp = pd.DataFrame(rows)
    print("\n[COMPARISON TABLE]")
    display(df_cmp.sort_values(["split", "reward_mean_per_step"], ascending=[True, False]).reset_index(drop=True))

    print("\n[COMPARISON PLOTS] test")
    plot_policy_comparison_inline(df_cmp, split_name="test")
    print("[COMPARISON PLOTS] train")
    plot_policy_comparison_inline(df_cmp, split_name="train")

    print("[DONE] Displayed sampled trajectories and comparison plots inline.")

if __name__ == "__main__":
    main()
