import torch
import random
from .DDQNscheduler import GRUQNetwork
from .train_environment import EnvConfig, WindowStore, RUWindowEnv

from typing import Dict, List
import numpy as np

@torch.no_grad()
def evaluate_on_windows(
    q_net: GRUQNetwork,
    store: WindowStore,
    test_indices: List[int],
    env_cfg: EnvConfig,
    device: torch.device,
    max_eval_episodes: int = 200,
) -> Dict[str, float]:
    q_net.eval()

    if len(test_indices) == 0:
        return {
            "eval_reward_mean": float("nan"),
            "eval_best_action_ratio": float("nan"),
            "eval_handover_attempt_per_step": float("nan"),
            "n_episodes": 0,
        }

    idxs = list(test_indices)
    random.shuffle(idxs)
    idxs = idxs[:max_eval_episodes]

    returns, best_ratios, ho_rates = [], [], []
    for ep_idx in idxs:
        env = RUWindowEnv(store, [ep_idx], env_cfg, training=False)
        obs = env.reset(ep_idx=ep_idx)
        h = q_net.init_hidden(1, device)

        ep_ret, ep_best, ep_ho, steps = 0.0, 0.0, 0.0, 0
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).to(device).view(1, -1)
            q_values, h = q_net(obs_t, h)
            act = int(torch.argmax(q_values, dim=-1).item())

            obs, r, done, info = env.step(act)
            ep_ret += float(r)
            ep_best += float(info["is_best_action"])
            ep_ho += float(info["handover_attempt"])
            steps += 1

        returns.append(ep_ret / max(steps, 1))
        best_ratios.append(ep_best / max(steps, 1))
        ho_rates.append(ep_ho / max(steps, 1))

    return {
        "eval_reward_mean": float(np.mean(returns)) if returns else float("nan"),
        "eval_best_action_ratio": float(np.mean(best_ratios)) if best_ratios else float("nan"),
        "eval_handover_attempt_per_step": float(np.mean(ho_rates)) if ho_rates else float("nan"),
        "n_episodes": int(len(returns)),
    }
