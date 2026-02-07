from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics
from minigrid.wrappers import ImgObsWrapper
import torch
import torchrl
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import (
    TransformedEnv,
    Compose,
    StepCounter,
    DTypeCastTransform,
)
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from tensordict.nn import TensorDictModule, TensorDictSequential


MAP_DEF = [
    "######",
    "#s...#",
    "#.##.#",
    "#.#..#",
    "#..#g#",
    "######",
]
ENV_ID = "MazeFromText-v0"

register(
    id=ENV_ID,
    entry_point="maze_from_text:MazeFromTextEnv",
    kwargs={"map_def": MAP_DEF},
)

DEVICE = torch.device("mps")
CHECKPOINT_DIR = "models"
ACTOR_PATH = Path(CHECKPOINT_DIR) / "actor.pt"
CRITIC_PATH = Path(CHECKPOINT_DIR) / "critic.pt"


def make_env(render_mode="rgb_array", max_steps=200):
    env = gym.make(ENV_ID, render_mode=render_mode, max_steps=max_steps)
    env = RecordEpisodeStatistics(env)
    env = ImgObsWrapper(env)
    env = FlattenObservation(env)

    base = GymWrapper(env, device=torch.device("cpu"), categorical_action_encoding=True)
    return TransformedEnv(
        base,
        Compose(
            [
                DTypeCastTransform(
                    dtype_in=torch.uint8,
                    dtype_out=torch.float32,
                    in_keys=["observation"],
                ),
                StepCounter(max_steps=max_steps),
            ]
        ),
    )


def build_model(obs_key, n_actions, n_hidden):
    linear = TensorDictModule(
        torch.nn.LazyLinear(out_features=n_hidden),
        in_keys=[obs_key],
        out_keys=["features"],
    )

    actor_head = TensorDictModule(
        torch.nn.LazyLinear(out_features=n_actions),
        in_keys=["features"],
        out_keys=["logits"],
    )

    actor = ProbabilisticActor(
        TensorDictSequential(linear, actor_head),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    critic_head = TensorDictModule(
        torch.nn.LazyLinear(out_features=1),
        in_keys=["features"],
        out_keys=["value"],
    )
    critic = ValueOperator(
        TensorDictSequential(linear, critic_head, selected_out_keys=["value"]),
        in_keys=[obs_key],
        out_keys=["state_value"],
    )

    return actor, critic


def initialize_model(env):
    n_actions = env.action_space.n
    actor, critic = build_model(
        obs_key="observation",
        n_actions=n_actions,
        n_hidden=128,
    )

    dummy_td = env.reset()
    with torch.no_grad():
        actor(dummy_td)
        critic(dummy_td)

    return actor, critic


def save_models(actor, critic, actor_path=ACTOR_PATH, critic_path=CRITIC_PATH):
    actor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)


def load_models(actor_path=ACTOR_PATH, critic_path=CRITIC_PATH):
    env = make_env(render_mode="rgb_array")
    actor, critic = initialize_model(env)
    actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
    return actor, critic


FRAMES_PER_BATCH = 2048
NUM_UPDATES = 10
BATCH_SIZE = 16
PPO_EPOCHS = 10
NUM_ENVS = 4
ENTROPY_COEFF = 0.0


def train(n_envs=4):
    env = make_env(render_mode="rgb_array")
    actor, critic = initialize_model(env)
    actor.to(DEVICE)
    critic.to(DEVICE)

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=3e-4,
    )

    envs = torchrl.envs.ParallelEnv(num_workers=n_envs, create_env_fn=make_env)
    collector = SyncDataCollector(
        envs,
        policy=actor,
        frames_per_batch=FRAMES_PER_BATCH,
        device=DEVICE,
        total_frames=-1,
    )

    storage = LazyTensorStorage(FRAMES_PER_BATCH)
    sampler = SamplerWithoutReplacement()
    replay_buffer = ReplayBuffer(
        storage=storage, sampler=sampler, batch_size=BATCH_SIZE
    )

    for update in range(NUM_UPDATES):
        batch = next(iter(collector))
        replay_buffer.extend(batch)

        for epoch in range(PPO_EPOCHS):
            for i, rb_batch in enumerate(replay_buffer):
                loss = loss_module(rb_batch.to(DEVICE))
                optimizer.zero_grad()
                loss_actor = (
                    loss["loss_objective"] - ENTROPY_COEFF * loss["loss_entropy"]
                )
                loss_actor.backward()
                loss["loss_critic"].backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    max_norm=0.5,
                )
                optimizer.step()

                if i == 0 and epoch == 0:
                    entries = {
                        "update": update,
                        "loss_actor": loss_actor.item(),
                        "loss_critic": loss["loss_critic"].item(),
                        "entropy": loss.get("entropy", None),
                        "kl_approx": loss.get("kl_approx", None),
                        "clip_fraction": loss.get("clip_fraction", None),
                        "ESS": loss.get("ESS", None),
                    }
                    output = ", ".join(
                        [f"{k}: {v:.4f}" for k, v in entries.items() if v is not None]
                    )
                    print(output)

    collector.shutdown()
    save_models(actor, critic)


def evaluate(episodes=5, is_dump=True):
    env = make_env(render_mode="human")
    actor, _ = load_models()
    actor = actor.to(torch.device("cpu"))
    actor.eval()

    for ep in range(episodes):
        td = env.reset()

        for _ in range(10000):
            step = td["step_count"].item()
            with torch.no_grad():
                td = actor(td)
            print(f"step {step}: action {td['action']}")

            td = env.step(td)

            if is_dump:
                base = getattr(env, "unwrapped", env)
                while hasattr(base, "env"):
                    base = base.env
                print(f"\n--- Episode {ep + 1} Step {step} ---")
                print(base.dump())

            time.sleep(0.1)

            td = td["next"]
            if td["done"].item():
                print(
                    f"Episode {ep + 1} finished after {step} steps, teminated={td['terminated'].item()}, truncated={td['truncated'].item()}"
                )
                break

    env.close()


if __name__ == "__main__":
    train(n_envs=NUM_ENVS)
    evaluate(episodes=1, is_dump=False)
