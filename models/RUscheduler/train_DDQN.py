from .Envconfig import DQNConfig
from .train_environment import EnvConfig, WindowStore, RUWindowEnv, VecEnv
from .DDQNscheduler import GRUQNetwork, ReplayBuffer
from .evaluate import evaluate_on_windows

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import deque
from typing import List, Optional
import time
from pathlib import Path
import math

GPU_ID = 3


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{GPU_ID}")
    return torch.device("cpu")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    if duration <= 0:
        return float(end)
    frac = min(max(step, 0) / float(duration), 1.0)
    return float(start + frac * (end - start))


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-dir", type=str, default="./data_ready/windows_ws4p00s_ov8")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--total-env-steps", type=int, default=1_500_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--replay-size", type=int, default=50_000)
    ap.add_argument("--learning-starts", type=int, default=2_000)
    ap.add_argument("--train-every", type=int, default=1)
    ap.add_argument("--gradient-steps", type=int, default=1)
    ap.add_argument("--target-update-interval", type=int, default=2_000)
    ap.add_argument("--max-grad-norm", type=float, default=10.0)
    ap.add_argument("--two-ru-penalty", type=float, default=2.0)
    ap.add_argument("--reward-snr-db", type=float, default=10.0)
    ap.add_argument("--epsilon-start", type=float, default=1.0)
    ap.add_argument("--epsilon-end", type=float, default=0.02)
    ap.add_argument("--epsilon-decay-steps", type=int, default=150_000)

    ap.add_argument("--hidden-size", type=int, default=8)

    ap.add_argument("--log-every", type=int, default=5_000)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--eval-max-episodes", type=int, default=200)
    ap.add_argument("--save-dir", type=str, default="./checkpoints_double_dqn_ru_windows4s_easiest")
    ap.add_argument("--save-every", type=int, default=50_000)

    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print("[ARGS] Ignoring unknown args (likely Jupyter):", unknown)

    seed_all(args.seed)
    device = get_device()
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[CUDA] GPU_ID={GPU_ID} | name: {torch.cuda.get_device_name(device)}")

    store = WindowStore(args.window_dir)
    print(f"[DATA] Loaded windows from: {Path(args.window_dir).resolve()}")
    print(f"[DATA] X shape: (N,L,6) = {store.X.shape} | dt={store.dt} | window_sec={store.window_sec}")

    forced_test_datasets = {"20260211_162702", "20260211_164725"}
    if "dataset" not in store.manifest.columns:
        raise ValueError("manifest.csv must contain a 'dataset' column to force test datasets.")

    all_indices = np.arange(len(store), dtype=np.int64)
    forced_mask = store.manifest["dataset"].isin(list(forced_test_datasets)).to_numpy(dtype=bool)
    forced_test_indices = all_indices[forced_mask].tolist()

    rng = np.random.RandomState(args.seed)
    remaining = all_indices[~forced_mask]
    rng.shuffle(remaining)

    total = len(all_indices)
    target_test = max(len(forced_test_indices), int(round(0.1 * total)))
    extra_needed = max(0, target_test - len(forced_test_indices))
    extra_test = remaining[:extra_needed].tolist()

    test_indices = sorted(forced_test_indices + extra_test)
    train_indices = sorted(list(set(all_indices.tolist()) - set(test_indices)))

    print(f"[SPLIT] windows N={total} => train={len(train_indices)} test={len(test_indices)}")

    env_cfg = EnvConfig(seg_len_ticks=20, obs_bin_ticks=20, obs_hist_bins=3, mask_value=0.0, handover_penalty=5e-1, inter_ru_phase_noise_std=math.pi / 8.0, two_ru_penalty=float(args.two_ru_penalty), noise_power=1e-3, reward_snr_db=float(args.reward_snr_db), max_episode_steps=0)
    dqn_cfg = DQNConfig(
        total_env_steps=int(args.total_env_steps),
        num_envs=int(args.num_envs),
        gamma=float(args.gamma),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        replay_size=int(args.replay_size),
        learning_starts=int(args.learning_starts),
        train_every=int(args.train_every),
        gradient_steps=int(args.gradient_steps),
        target_update_interval=int(args.target_update_interval),
        max_grad_norm=float(args.max_grad_norm),
        epsilon_start=float(args.epsilon_start),
        epsilon_end=float(args.epsilon_end),
        epsilon_decay_steps=int(args.epsilon_decay_steps),
    )

    train_envs = [RUWindowEnv(store, train_indices, env_cfg, training=True) for _ in range(dqn_cfg.num_envs)]
    venv = VecEnv(train_envs)
    obs = venv.reset()

    q_net = GRUQNetwork(obs_dim=venv.obs_dim, hidden_size=int(args.hidden_size), n_actions=venv.n_actions).to(device)
    target_net = GRUQNetwork(obs_dim=venv.obs_dim, hidden_size=int(args.hidden_size), n_actions=venv.n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)

    n_params = count_trainable_params(q_net)
    print(
        f"[MODEL] obs_dim={venv.obs_dim} hidden_size={q_net.hidden_size} "
        f"n_actions={q_net.n_actions} trainable_params={n_params}"
    )
    print("[TASK] Obs=masked RU gains from -400~-100ms (9D) + prev-conn onehot (3D), Reward=power@(k+1)-0.0005*handover.")

    optim = torch.optim.Adam(q_net.parameters(), lr=dqn_cfg.lr, eps=1e-5)
    replay = ReplayBuffer(capacity=dqn_cfg.replay_size, obs_dim=venv.obs_dim, hidden_size=q_net.hidden_size)
    os.makedirs(args.save_dir, exist_ok=True)

    h_env = q_net.init_hidden(dqn_cfg.num_envs, device).detach().cpu().numpy().astype(np.float32)

    ep_ret_running = np.zeros((dqn_cfg.num_envs,), dtype=np.float64)
    ep_best_running = np.zeros((dqn_cfg.num_envs,), dtype=np.float64)
    ep_ho_running = np.zeros((dqn_cfg.num_envs,), dtype=np.float64)
    ep_len_running = np.zeros((dqn_cfg.num_envs,), dtype=np.int64)

    recent_returns = deque(maxlen=100)
    recent_best = deque(maxlen=100)
    recent_handover = deque(maxlen=100)
    recent_losses = deque(maxlen=200)

    global_step = 0
    env_iter = 0
    t_start = time.time()
    last_target_sync = 0
    next_log_step = int(args.log_every)
    next_eval_step = int(args.eval_every)
    next_save_step = int(args.save_every)

    while global_step < dqn_cfg.total_env_steps:
        q_net.train()

        epsilon = linear_schedule(global_step, dqn_cfg.epsilon_start, dqn_cfg.epsilon_end, dqn_cfg.epsilon_decay_steps)

        obs_t = torch.from_numpy(obs).to(device)
        h_t = torch.from_numpy(h_env).to(device)
        with torch.no_grad():
            q_values, h_next_t = q_net(obs_t, h_t)
            greedy_actions = torch.argmax(q_values, dim=-1).cpu().numpy()

        random_actions = np.random.randint(0, venv.n_actions, size=dqn_cfg.num_envs, dtype=np.int64)
        explore_mask = (np.random.rand(dqn_cfg.num_envs) < epsilon)
        actions = np.where(explore_mask, random_actions, greedy_actions).astype(np.int64)

        next_obs, rewards, dones, infos = venv.step(actions)

        h_next_np = h_next_t.detach().cpu().numpy().astype(np.float32)
        # Reset hidden state for envs that auto-reset episode
        for i in range(dqn_cfg.num_envs):
            if bool(infos[i].get("episode_reset", False)):
                h_next_np[i, :] = 0.0

        for i in range(dqn_cfg.num_envs):
            ep_ret_running[i] += float(rewards[i])
            ep_best_running[i] += float(infos[i]["is_best_action"])
            ep_ho_running[i] += float(infos[i]["handover_attempt"])
            ep_len_running[i] += 1
            if dones[i]:
                recent_returns.append(float(ep_ret_running[i] / max(ep_len_running[i], 1)))
                recent_best.append(float(ep_best_running[i] / max(ep_len_running[i], 1)))
                recent_handover.append(float(ep_ho_running[i] / max(ep_len_running[i], 1)))
                ep_ret_running[i] = 0.0
                ep_best_running[i] = 0.0
                ep_ho_running[i] = 0.0
                ep_len_running[i] = 0

        replay.add_batch(obs, h_env, actions, rewards, next_obs, h_next_np, dones.astype(np.float32))

        obs = next_obs
        h_env = h_next_np
        env_iter += 1
        global_step += dqn_cfg.num_envs

        if (global_step >= dqn_cfg.learning_starts) and (len(replay) >= dqn_cfg.batch_size) and (env_iter % dqn_cfg.train_every == 0):
            for _ in range(dqn_cfg.gradient_steps):
                b_obs, b_h, b_act, b_rew, b_next_obs, b_next_h, b_done = replay.sample(dqn_cfg.batch_size, device)

                q_pred_all, _ = q_net(b_obs, b_h)
                q_pred = q_pred_all.gather(1, b_act.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    next_online_q, _ = q_net(b_next_obs, b_next_h)
                    next_actions = next_online_q.argmax(dim=-1, keepdim=True)
                    next_target_q, _ = target_net(b_next_obs, b_next_h)
                    next_q = next_target_q.gather(1, next_actions).squeeze(1)
                    target = b_rew + dqn_cfg.gamma * (1.0 - b_done) * next_q

                loss = F.smooth_l1_loss(q_pred, target)
                recent_losses.append(float(loss.item()))

                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), dqn_cfg.max_grad_norm)
                optim.step()

        if (global_step - last_target_sync) >= dqn_cfg.target_update_interval:
            target_net.load_state_dict(q_net.state_dict())
            last_target_sync = global_step

        if global_step >= next_log_step:
            elapsed_min = (time.time() - t_start) / 60.0
            rew_mean = float(np.mean(recent_returns)) if recent_returns else float("nan")
            best_mean = float(np.mean(recent_best)) if recent_best else float("nan")
            ho_mean = float(np.mean(recent_handover)) if recent_handover else float("nan")
            loss_mean = float(np.mean(recent_losses)) if recent_losses else float("nan")
            print(
                f"[step {global_step:9d}] eps={epsilon:.3f} buffer={len(replay):6d} loss={loss_mean:.4f} "
                f"train_reward/step={rew_mean:.4f} train_best_action_ratio={best_mean:.4f} train_handover/step={ho_mean:.5f} | t={elapsed_min:.1f} min"
            )
            next_log_step += int(args.log_every)

        if (args.eval_every > 0) and (global_step >= next_eval_step):
            m = evaluate_on_windows(q_net=q_net, store=store, test_indices=test_indices, env_cfg=env_cfg, device=device, max_eval_episodes=int(args.eval_max_episodes))
            elapsed_min = (time.time() - t_start) / 60.0
            print(
                f"[EVAL step {global_step:9d}] reward/step={m['eval_reward_mean']:.4f} "
                f"best_action_ratio={m['eval_best_action_ratio']:.4f} handover/step={m['eval_handover_attempt_per_step']:.5f} "
                f"n_ep={m['n_episodes']} | t={elapsed_min:.1f} min"
            )
            next_eval_step += int(args.eval_every)

        if (args.save_every > 0) and (global_step >= next_save_step):
            ckpt = os.path.join(args.save_dir, f"double_dqn_ru_easiest_step{global_step:09d}.pt")
            torch.save(
                {
                    "global_step": global_step,
                    "env_iter": env_iter,
                    "q_net": q_net.state_dict(),
                    "target_net": target_net.state_dict(),
                    "optim": optim.state_dict(),
                    "env_cfg": env_cfg.__dict__,
                    "dqn_cfg": dqn_cfg.__dict__,
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                    "window_dir": str(Path(args.window_dir).resolve()),
                    "gain_scales_in_xnpz": store.gain_scales,
                    "gain_match_enable": store.gain_match_enable,
                    "args": vars(args),
                    "GPU_ID": GPU_ID,
                    "model_meta": {
                        "obs_dim": venv.obs_dim,
                        "hidden_size": q_net.hidden_size,
                        "n_actions": q_net.n_actions,
                        "n_params": n_params,
                        "algo": "double_dqn_easiest_gru",
                        "task_alignment": "obs[k-3,k-2,k-1] masked + conn_onehot -> reward_power@k+1 - 0.0005*handover",
                    },
                },
                ckpt,
            )
            print(f"[SAVE] {ckpt}")
            next_save_step += int(args.save_every)

    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
