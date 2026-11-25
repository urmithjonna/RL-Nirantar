"""Learned controller: behaviour cloning -> DAgger -> PPO, with an MLP or a GRU (deep sequence) policy network.
Written from notebooks/05_imitation.ipynb. Settings fixed in results/FINAL_SPEC.md section 5.

Data layout for cloning: F = states [N, 23]; IDX = [N, 8] row numbers in F of the last 8 states (oldest first, the first state
repeated at the start of an episode, exactly as the simulator stacks them); y = expert action; hold = cloning hold-out flag.
MLP input = F[IDX[:, -1]] (current state); GRU input = F[IDX] (last 2 hours)."""
import copy, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sim import StreetEnv, N_OBS, load_bank, load_cfg, evaluate
from controllers import expert_label

K_HIST = 8                          # GRU: last 8 states = 2 hours
NETS = {"MLP": 1, "GRU": K_HIST}    # network -> states stacked in the observation
STEPS, EVAL_EVERY, PPO_LR = 1_000_000, 100_000, 1e-4
N_ROUNDS, ROLL_WINDOWS, LABEL_STRIDE = 2, 300, 8


class GRUExtractor(BaseFeaturesExtractor):
    """Last K states [K x 23] -> GRU (64 hidden) -> last hidden state, concatenated with the current state (64 + 23 = 87)."""
    def __init__(self, observation_space: spaces.Box, k=K_HIST, n_in=N_OBS, hidden=64):
        super().__init__(observation_space, features_dim=hidden + n_in)
        self.k, self.n_in = k, n_in
        self.gru = nn.GRU(n_in, hidden, batch_first=True)

    def forward(self, obs):
        x = obs.view(-1, self.k, self.n_in)
        _, h = self.gru(x)
        return torch.cat([h[-1], x[:, -1]], dim=1)


def make_ppo(env, seed, net, lr=PPO_LR):
    kw = dict(net_arch=[128, 128])
    if net == "GRU":
        kw.update(features_extractor_class=GRUExtractor, features_extractor_kwargs=dict(k=K_HIST, n_in=N_OBS, hidden=64))
    return PPO("MlpPolicy", env, learning_rate=lr, n_steps=2048, batch_size=256, n_epochs=10, gae_lambda=0.95, clip_range=0.2,
               ent_coef=0.01, gamma=0.995, policy_kwargs=kw, seed=seed, device="cpu", verbose=0)


def env_kw(net):
    return dict(shield=True, stack=NETS[net])


# ---------- behaviour cloning ----------
def stack_index(n_steps, offset, k=K_HIST):
    t = np.arange(n_steps)
    return (offset + np.maximum(t[:, None] - (k - 1) + np.arange(k)[None, :], 0)).astype(np.int64)


def clone(bank, train_eps, cfg, net, seed, F, IDX, y, hold, max_epochs=30, batch=1024):
    """Cross-entropy cloning into a fresh PPO policy (seed `seed`, the same initialisation PPO starts from); Adam 1e-3, batch 1024,
    epoch with the lowest hold-out cross-entropy kept. Runs on the GPU when available."""
    torch.manual_seed(seed); torch.backends.cudnn.deterministic = True
    model = make_ppo(StreetEnv(bank, train_eps, cfg, train=True, seed=seed, **env_kw(net)), seed, net)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol = model.policy.to(dev); pol.set_training_mode(True)
    Ft, It, yt = torch.tensor(F, device=dev), torch.tensor(IDX if net == "GRU" else IDX[:, -1:], device=dev), torch.tensor(y, device=dev)
    X = lambda j: Ft[It[j]].reshape(len(j), -1)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3); rng = np.random.default_rng(seed)
    tr, ho = np.where(~hold)[0], torch.tensor(np.where(hold)[0], device=dev)
    best, best_state, log = np.inf, None, []
    for epoch in range(1, max_epochs + 1):
        perm = torch.tensor(rng.permutation(tr), device=dev)
        for b in range(0, len(perm), batch):
            j = perm[b:b + batch]
            loss = -pol.get_distribution(X(j)).log_prob(yt[j]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ce, acc = 0.0, 0.0
            for b in range(0, len(ho), 65536):
                j = ho[b:b + 65536]; dist = pol.get_distribution(X(j))
                ce += float(-dist.log_prob(yt[j]).sum()); acc += float((dist.distribution.probs.argmax(1) == yt[j]).sum())
            ce, acc = ce / len(ho), acc / len(ho)
        log.append({"epoch": epoch, "holdout_cross_entropy": ce, "holdout_accuracy": acc})
        if ce < best:
            best, best_state = ce, copy.deepcopy(pol.state_dict())
    pol.load_state_dict(best_state); pol.set_training_mode(False); model.policy.to("cpu")
    return model, pd.DataFrame(log)


# ---------- DAgger ----------
def draw_homes(eps, seed):
    rng = np.random.default_rng(seed)
    return [{**e, "homes_idx": sorted(rng.choice(e["pool_idx"], 10, replace=False).tolist())} for e in eps]


def rollout(model, bank, eps, cfg, net, seed, stride=LABEL_STRIDE):
    """The learner's own trajectories (sampled actions, shield off, random start charge 30-90%). Every `stride`-th state is kept
    with what is needed to label it (episode, step, battery energy) and the learner's greedy action (for the disagreement rate)."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    envs = [StreetEnv(bank, [e], cfg, train=False, shield=False, stack=K_HIST) for e in eps]
    obs = []
    for env in envs:
        env.reset(options={"episode": 0}); env.soc = rng.uniform(0.3, 0.9) * cfg.batt_kwh; obs.append(env.restart_history())
    obs = np.stack(obs); cut = (lambda o: o) if net == "GRU" else (lambda o: o[:, -N_OBS:])
    rec = {"obs": [], "ep": [], "t": [], "soc": [], "greedy": []}
    for t in range(envs[0].T):
        if t % stride == 0:
            rec["obs"].append(obs.copy()); rec["ep"].append(np.arange(len(envs))); rec["t"].append(np.full(len(envs), t))
            rec["soc"].append(np.array([e.soc for e in envs])); rec["greedy"].append(model.predict(cut(obs), deterministic=True)[0])
        act = model.predict(cut(obs), deterministic=False)[0]
        for i, env in enumerate(envs):
            obs[i] = env.step(int(act[i]))[0]
    return {k: np.concatenate(v) for k, v in rec.items()}


def label_chunk(bank_path, cfg, items):
    """items: [(episode, [(t, soc), ...]), ...] -> expert actions in order (runs in a joblib worker)."""
    bank = load_bank(bank_path); out = []
    for ep, ts in items:
        env = StreetEnv(bank, [ep], cfg, train=False, shield=False); env.reset(options={"episode": 0})
        out += [expert_label(env, int(t), float(s), cfg) for t, s in ts]
    return out


# ---------- PPO ----------
class ValCallback(BaseCallback):
    """Validation (deterministic, shield on) at step 0 and every 100k steps; the best mean reward is kept as best_model."""
    def __init__(self, meta, bank, val, cfg, kw, out):
        super().__init__(); self.meta, self.bank, self.val, self.cfg, self.kw, self.out = meta, bank, val, cfg, kw, out
        self.best, self.rows, self.t0 = -np.inf, [], time.time()

    def check(self):
        ep = pd.DataFrame(evaluate(self.model, self.bank, self.val, self.cfg, self.kw))
        row = {**self.meta, "step": self.num_timesteps, "minutes": (time.time() - self.t0) / 60, **ep.drop(columns="episode").mean().to_dict()}
        self.rows.append(row); pd.DataFrame(self.rows).to_csv(self.out / "trajectory.csv", index=False)
        if row["reward_per_week"] > self.best:
            self.best = row["reward_per_week"]; self.model.save(self.out / "best_model")
            ep.to_csv(self.out / "best_val_per_episode.csv", index=False)
            (self.out / "best.json").write_text(json.dumps({"step": self.num_timesteps, "val_reward_per_week": self.best}))

    def _on_training_start(self):
        self.check()

    def _on_step(self):
        if self.num_timesteps % EVAL_EVERY == 0:
            self.check()
        return True


def train_ppo(root, net, seed, steps=STEPS):
    """PPO from the frozen imitation policy of (net, seed). Writes models/rl/{net}_s{seed}/."""
    torch.set_num_threads(1)
    root = Path(root); cfg = load_cfg(root)
    bank = load_bank(root / "data" / "processed" / "bank_trainval.pkl")
    E = json.loads((root / "data" / "processed" / "episodes.json").read_text())
    out = root / "models" / "rl" / f"{net}_s{seed}"; out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        return str(out)
    model = make_ppo(StreetEnv(bank, E["train"], cfg, train=True, seed=seed, **env_kw(net)), seed, net)
    src = PPO.load(root / "models" / "imitation" / f"{net}_s{seed}" / "chosen.zip", device="cpu")
    model.policy.load_state_dict(src.policy.state_dict())
    meta = {"net": net, "seed": seed}
    t0 = time.time()
    model.learn(steps, callback=ValCallback(meta, bank, E["val"], cfg, env_kw(net), out))
    (out / "done.json").write_text(json.dumps({**meta, "steps": steps, "minutes": (time.time() - t0) / 60}))
    return str(out)
