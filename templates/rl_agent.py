"""
NM i AI 2026 — Reinforcement Learning Agent Template
=====================================================
Bruker Gymnasium + Stable-Baselines3 PPO for a trene en RL-agent.

Slik bruker du:
1. Definer miljoet ditt i CustomEnv-klassen (observation, action, reward)
2. Juster hyperparametre i train()-funksjonen
3. Kjor: python rl_agent.py

TODO-markerte steder ma tilpasses for den spesifikke oppgaven.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import json


# === CUSTOM ENVIRONMENT ===
# TODO: Tilpass dette miljoet til den spesifikke oppgaven

class CustomEnv(gym.Env):
    """
    Custom Gymnasium-miljo for konkurransen.

    Slik fungerer RL:
    - Agenten OBSERVERER miljoet (observation)
    - Agenten VELGER en handling (action)
    - Miljoet gir BELONNING (reward) og ny tilstand
    - Agenten laerer a maksimere total belonning over tid
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # TODO: Definer observation space — hva agenten kan "se"
        # Eksempler:
        #   Box: kontinuerlige verdier (posisjon, hastighet, etc.)
        #   Discrete: en av N kategorier
        #   MultiBinary: flere ja/nei-verdier
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),  # TODO: Endre til riktig storrelse
            dtype=np.float32
        )

        # TODO: Definer action space — hva agenten kan gjore
        # Eksempler:
        #   Discrete(4): velg en av 4 handlinger (opp/ned/venstre/hoyre)
        #   Box: kontinuerlig handling (f.eks. styrevinkel)
        self.action_space = spaces.Discrete(4)  # TODO: Endre til riktig antall

        # TODO: Initialiser miljo-state
        self.state = None
        self.step_count = 0
        self.max_steps = 1000  # TODO: Sett maks antall steg per episode

    def reset(self, seed=None, options=None):
        """Tilbakestill miljoet til starttilstand."""
        super().reset(seed=seed)

        # TODO: Sett opp starttilstand
        self.state = np.zeros(10, dtype=np.float32)  # TODO: Riktig starttilstand
        self.step_count = 0

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        """Utfor en handling og returner resultat."""
        self.step_count += 1

        # TODO: Oppdater miljoet basert pa handlingen
        # Eksempel:
        # if action == 0: self.agent_pos[1] -= 1  # opp
        # if action == 1: self.agent_pos[1] += 1  # ned
        # if action == 2: self.agent_pos[0] -= 1  # venstre
        # if action == 3: self.agent_pos[0] += 1  # hoyre

        # TODO: Beregn belonning
        # Tips: Belonningen er det VIKTIGSTE a fa riktig!
        # - Positiv belonning for onskede handlinger
        # - Negativ belonning for uonskede handlinger
        # - Sma steg-straff (-0.01) for a oppmuntre rask losning
        reward = 0.0  # TODO: Beregn belonning

        # TODO: Sjekk om episoden er ferdig
        terminated = False  # Oppgaven er lost/tapt
        truncated = self.step_count >= self.max_steps  # Timeout

        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Konverter intern state til observation for agenten."""
        # TODO: Returner det agenten kan "se"
        return self.state.copy()

    def render(self):
        """Visualiser miljoet (valgfritt)."""
        if self.render_mode == "human":
            # TODO: Print eller vis miljoet
            print(f"Step {self.step_count}: state={self.state[:3]}...")


# === WRAPPER FOR EKSTERN API ===
# Bruk denne hvis miljoet kjorer via API (som grocery bot)

class APIEnv(gym.Env):
    """
    Wrapper som kobler Gymnasium til et eksternt API-miljo.
    Nyttig nol oppgaven har en server du kommuniserer med.
    """

    metadata = {"render_modes": []}

    def __init__(self, api_url: str, api_key: str):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key

        # TODO: Definer spaces basert pa API-spesifikasjonen
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # TODO: Kall API for a starte nytt spill/episode
        # response = requests.post(f"{self.api_url}/reset", headers={"Authorization": self.api_key})
        # state = response.json()
        observation = np.zeros(10, dtype=np.float32)
        return observation, {}

    def step(self, action):
        # TODO: Send action til API, motta ny state
        # response = requests.post(f"{self.api_url}/step", json={"action": int(action)})
        # data = response.json()
        observation = np.zeros(10, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated, {}


# === CUSTOM CALLBACK ===
# Logger trenings-fremgang

class ProgressCallback(BaseCallback):
    """Skriver ut trenings-fremgang underveis."""

    def __init__(self, log_interval=1000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []

    def _on_step(self):
        if self.n_calls % self.log_interval == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                mean_length = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                print(f"  Steg {self.n_calls}: snitt belonning={mean_reward:.2f}, snitt lengde={mean_length:.0f}")
        return True


# === TRENING ===

def train(
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    save_path: str = "models/rl_agent",
):
    """
    Tren en PPO-agent.

    Args:
        total_timesteps: Totalt antall treningssteg (mer = bedre, men tregere)
        n_envs: Antall parallelle miljoer (raskere trening)
        save_path: Hvor modellen lagres
    """
    print("=== RL Agent — Trening ===")

    # Lag parallelle miljoer for raskere trening
    def make_env():
        def _init():
            env = CustomEnv()  # TODO: Bytt til APIEnv hvis relevant
            env = Monitor(env)
            return env
        return _init

    env = DummyVecEnv([make_env() for _ in range(n_envs)])

    # TODO: Juster hyperparametre
    # PPO er en god default for de fleste RL-oppgaver
    model = PPO(
        "MlpPolicy",          # Neuralt nettverk-policy (MLP = fully connected)
        env,
        learning_rate=3e-4,    # Laeringsrate — senk hvis ustabil trening
        n_steps=2048,          # Steg per oppdatering
        batch_size=64,         # Batch-storrelse
        n_epochs=10,           # Epoker per oppdatering
        gamma=0.99,            # Diskonteringsfaktor (0.99 = langsiktig, 0.9 = kortsiktig)
        gae_lambda=0.95,       # GAE lambda
        clip_range=0.2,        # PPO clip range
        ent_coef=0.01,         # Entropi-koeffisient (hoyere = mer utforsking)
        verbose=1,
        tensorboard_log="./logs/",
    )

    # Eval callback — evaluerer agenten underveis
    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best/",
        log_path=f"{save_path}/eval_logs/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Tren!
    print(f"Trener i {total_timesteps} steg med {n_envs} parallelle miljoer...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, ProgressCallback()],
        progress_bar=True,
    )

    # Lagre modell
    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}/final_model")
    print(f"Modell lagret til {save_path}/final_model")

    return model


# === EVALUERING ===

def evaluate(model_path: str, n_episodes: int = 10):
    """Evaluer en trent modell."""
    print(f"=== Evaluerer {model_path} ===")

    model = PPO.load(model_path)
    env = CustomEnv()

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        print(f"  Episode {ep+1}: belonning = {total_reward:.2f}")

    print(f"\nSnitt belonning: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    return rewards


# === KJOR ===

if __name__ == "__main__":
    # Tren agent
    model = train(total_timesteps=100_000, n_envs=4)

    # Evaluer
    evaluate("models/rl_agent/final_model", n_episodes=10)
