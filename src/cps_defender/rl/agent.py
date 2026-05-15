"""
Deep Q-Network (DQN) agent — pure NumPy implementation.

Design rationale for numpy-only:
  • Eliminates PyTorch/TF as a hard dependency (saves ~2 GB install).
  • The state/action spaces here are tiny (8-dim state, 5 actions), so a
    2-hidden-layer MLP trained with SGD is perfectly sufficient.
  • Makes the learning dynamics transparent and reproducible.

Architecture:
  Input(8) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(5, linear)

Training:
  • Experience replay buffer (FIFO, configurable size).
  • Target network updated every `target_update_freq` steps.
  • ε-greedy exploration with exponential decay.
  • Mini-batch SGD with MSE loss on Bellman targets.

Usage:
    agent = DQNAgent(obs_dim=8, n_actions=5)
    obs = env.reset()
    action = agent.select_action(obs)
    result = env.step(action)
    agent.remember(obs, action, result.reward, result.observation, result.done)
    agent.learn()
"""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Tiny numpy MLP ────────────────────────────────────────────────────────────

class NumpyMLP:
    """
    2-hidden-layer feedforward network.
    Forward: ReLU → ReLU → linear output.
    Backward: vanilla SGD on MSE.
    """

    def __init__(self, layer_sizes: List[int], lr: float = 1e-3, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.weights: List[np.ndarray] = []
        self.biases:  List[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He initialisation for ReLU layers
            scale = np.sqrt(2.0 / fan_in)
            W = rng.normal(0, scale, (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Returns (output, list_of_activations_for_backprop)."""
        activations = [x.astype(np.float32)]
        h = x.astype(np.float32)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            if i < len(self.weights) - 1:
                h = np.maximum(0, z)   # ReLU
            else:
                h = z                  # linear output
            activations.append(h)
        return h, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        out, _ = self.forward(x)
        return out

    def update(self, x: np.ndarray, targets: np.ndarray) -> float:
        """One SGD step. Returns MSE loss."""
        out, activations = self.forward(x)
        loss = float(np.mean((out - targets) ** 2))

        # Backprop
        delta = 2.0 * (out - targets) / targets.shape[0]   # MSE gradient
        for i in reversed(range(len(self.weights))):
            h_in = activations[i]
            dW = h_in.T @ delta
            db = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (activations[i] > 0)  # ReLU grad
            self.weights[i] -= self.lr * np.clip(dW, -1.0, 1.0)
            self.biases[i]  -= self.lr * np.clip(db, -1.0, 1.0)

        return loss

    def copy_from(self, other: "NumpyMLP") -> None:
        self.weights = [w.copy() for w in other.weights]
        self.biases  = [b.copy() for b in other.biases]

    def save(self, path: str) -> None:
        data = {
            f"W{i}": w for i, w in enumerate(self.weights)
        }
        data.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez(path, lr=self.lr, **data)

    @classmethod
    def load(cls, path: str, layer_sizes: List[int]) -> "NumpyMLP":
        data = np.load(path + ".npz", allow_pickle=True)
        net = cls(layer_sizes, lr=float(data["lr"]))
        net.weights = [data[f"W{i}"] for i in range(len(net.weights))]
        net.biases  = [data[f"b{i}"] for i in range(len(net.biases))]
        return net


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self._buf: Deque[Tuple] = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done) -> None:
        self._buf.append((
            np.asarray(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int) -> Optional[Tuple]:
        if len(self._buf) < batch_size:
            return None
        indices = np.random.choice(len(self._buf), batch_size, replace=False)
        batch = [self._buf[i] for i in indices]
        obs_b      = np.stack([t[0] for t in batch])
        actions_b  = np.array([t[1] for t in batch], dtype=np.int32)
        rewards_b  = np.array([t[2] for t in batch], dtype=np.float32)
        next_obs_b = np.stack([t[3] for t in batch])
        dones_b    = np.array([t[4] for t in batch], dtype=np.float32)
        return obs_b, actions_b, rewards_b, next_obs_b, dones_b

    def __len__(self) -> int:
        return len(self._buf)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN with experience replay + target network.
    Operates in the CPSEnvironment action/observation space.
    """

    def __init__(
        self,
        obs_dim: int = 8,
        n_actions: int = 5,
        hidden_sizes: List[int] = (64, 64),
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        memory_size: int = 10_000,
        target_update_freq: int = 50,
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.seed = seed

        np.random.seed(seed)

        layer_sizes = [obs_dim, *hidden_sizes, n_actions]
        self.online_net = NumpyMLP(layer_sizes, lr=lr, seed=seed)
        self.target_net = NumpyMLP(layer_sizes, lr=lr, seed=seed + 1)
        self.target_net.copy_from(self.online_net)

        self.memory = ReplayBuffer(memory_size)

        self._step = 0
        self._episode = 0
        self.losses: List[float] = []
        self.episode_rewards: List[float] = []

    # ── Policy ────────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_vals = self.online_net.predict(obs.reshape(1, -1))[0]
        return int(np.argmax(q_vals))

    # ── Memory & learning ─────────────────────────────────────────────────────

    def remember(self, obs, action, reward, next_obs, done) -> None:
        self.memory.push(obs, action, reward, next_obs, done)

    def learn(self) -> Optional[float]:
        """Sample a mini-batch and perform one gradient step. Returns loss."""
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None

        obs_b, actions_b, rewards_b, next_obs_b, dones_b = batch

        # Q(s,a) from online network
        q_online = self.online_net.predict(obs_b)  # (B, n_actions)

        # Target: r + γ * max_a Q_target(s', a) * (1 - done)
        q_next = self.target_net.predict(next_obs_b)  # (B, n_actions)
        max_q_next = q_next.max(axis=1)               # (B,)
        td_targets = rewards_b + self.gamma * max_q_next * (1.0 - dones_b)

        # Only update the Q-value for the taken action
        targets = q_online.copy()
        targets[np.arange(len(actions_b)), actions_b] = td_targets

        loss = self.online_net.update(obs_b, targets)
        self.losses.append(loss)
        self._step += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Sync target network
        if self._step % self.target_update_freq == 0:
            self.target_net.copy_from(self.online_net)

        return loss

    # ── Episode tracking ──────────────────────────────────────────────────────

    def end_episode(self, episode_reward: float) -> None:
        self.episode_rewards.append(episode_reward)
        self._episode += 1

    def get_stats(self) -> Dict:
        recent_losses = self.losses[-100:]
        recent_rewards = self.episode_rewards[-20:]
        return {
            "step": self._step,
            "episode": self._episode,
            "epsilon": self.epsilon,
            "mean_loss": float(np.mean(recent_losses)) if recent_losses else 0.0,
            "mean_reward_20ep": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            "buffer_size": len(self.memory),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        base = path.replace(".npz", "")
        self.online_net.save(base + "_online")
        self.target_net.save(base + "_target")
        np.savez(
            base + "_meta.npz",
            epsilon=self.epsilon,
            step=self._step,
            episode=self._episode,
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
        )
        logger.info("DQNAgent saved → %s", base)

    @classmethod
    def load(cls, path: str, **kwargs) -> "DQNAgent":
        base = path.replace(".npz", "")
        meta = np.load(base + "_meta.npz")
        agent = cls(
            obs_dim=int(meta["obs_dim"]),
            n_actions=int(meta["n_actions"]),
            **kwargs,
        )
        n_layers = len(agent.online_net.weights)
        layer_sizes = (
            [agent.obs_dim]
            + kwargs.get("hidden_sizes", [64, 64])
            + [agent.n_actions]
        )
        agent.online_net = NumpyMLP.load(base + "_online", layer_sizes)
        agent.target_net = NumpyMLP.load(base + "_target", layer_sizes)
        agent.epsilon = float(meta["epsilon"])
        agent._step    = int(meta["step"])
        agent._episode = int(meta["episode"])
        logger.info("DQNAgent loaded ← %s (ε=%.3f)", base, agent.epsilon)
        return agent


# ── Training loop ─────────────────────────────────────────────────────────────

def train_agent(
    env,
    agent: DQNAgent,
    max_episodes: int = 500,
    log_every: int = 50,
) -> DQNAgent:
    """
    Standard DQN training loop.
    Returns the trained agent.
    """
    logger.info("Starting DQN training: %d episodes", max_episodes)

    for ep in range(1, max_episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            result = env.step(action)
            agent.remember(obs, action, result.reward, result.observation, result.done)
            agent.learn()
            obs = result.observation
            ep_reward += result.reward
            done = result.done

        agent.end_episode(ep_reward)

        if ep % log_every == 0 or ep == max_episodes:
            stats = agent.get_stats()
            summary = env.episode_summary()
            logger.info(
                "Ep %4d | reward=%.2f | ε=%.3f | loss=%.4f | precision=%.2f | recall=%.2f",
                ep,
                summary["total_reward"],
                stats["epsilon"],
                stats["mean_loss"],
                summary["precision"],
                summary["recall"],
            )

    logger.info("Training complete. Final ε=%.4f", agent.epsilon)
    return agent
