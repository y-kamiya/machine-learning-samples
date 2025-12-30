from __future__ import annotations

import time

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics, TimeLimit
from minigrid.wrappers import ImgObsWrapper
import torch
import torchrl
from torch.torch_version import TorchVersion
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import TransformedEnv, Compose, StepCounter, DTypeCastTransform
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from tensordict.nn import TensorDictModule, TensorDictSequential

from maze_from_text import MazeFromTextEnv


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

DEVICE = torch.device("cpu")


def make_env(render_mode="rgb_array", max_steps=200, is_train=True):
    env = gym.make(ENV_ID, render_mode=render_mode)
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = ImgObsWrapper(env)
    env = FlattenObservation(env)

    base = GymWrapper(env, device=DEVICE)
    return TransformedEnv(base, Compose([
        DTypeCastTransform(dtype_in=torch.uint8, dtype_out=torch.float32, in_keys=["observation"]),
        StepCounter(),
    ]))


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

NUM_ENVS = 1
FRAMES_PER_BATCH = 2048
NUM_UPDATES = 100
BATCH_SIZE = 16
PPO_EPOCHS = 4

def train(total_timesteps=100000, n_envs=4):
    env = make_env(render_mode="rgb_array")
    n_actions = env.action_space.n

    actor, critic = build_model(
        obs_key="observation",
        n_actions=n_actions,
        n_hidden=128,
    )
    actor.to(DEVICE)
    critic.to(DEVICE)

    dummy_td = env.reset().to(DEVICE)
    with torch.no_grad():
        actor(dummy_td)
        critic(dummy_td)

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=0.2,
        entropy_bonus=0.01,
        value_loss_coef=0.5,
        gae_lambda=0.95,
        discount_factor=0.99,
        max_grad_norm=0.5,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=3e-4,
    )

    envs = torchrl.envs.ParallelEnv(num_workers=n_envs, create_env_fn=make_env)
    collector = SyncDataCollector(envs, policy=actor, frames_per_batch=FRAMES_PER_BATCH, device=DEVICE, total_frames=-1)

    storage = LazyTensorStorage(FRAMES_PER_BATCH)
    sampler = SamplerWithoutReplacement()
    replay_buffer = ReplayBuffer(storage=storage, sampler=sampler, batch_size=BATCH_SIZE)

    for update in range(NUM_UPDATES):
        batch = next(iter(collector))
        print(batch)
        import sys
        sys.exit()

        replay_buffer.extend(batch)

        for epoch in range(PPO_EPOCHS):
            for i, rb_batch in enumerate(replay_buffer):
                loss = loss_module(rb_batch.to(DEVICE))
                optimizer.zero_grad()
                loss.backward()
                loss["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    max_norm=0.5,
                )
                optimizer.step()

                if i == 0 and update % 10 == 0:
                    approx_kl = loss.get("approx_kl", None)
                    clipfrac = loss.get("clip_fraction", None)
                    print(f"Update {update}, loss: {loss['loss'].item():.3f}, approx_kl: {approx_kl}, clipfrac: {clipfrac}")

    collector.shutdown()


def evaluate(episodes=5, is_dump=True):
    env = make_env(render_mode="human")()

    for ep in range(episodes):
        td_reset = env.reset()
        done = False
        step = 0
        while True:
            action = env.rand_action(td_reset)
            print(f"step {step}: action {action['action']}")
            td = env.step(action)
            step += 1

            if is_dump:
                base = getattr(env, "unwrapped", env)
                while hasattr(base, "env"):
                    base = base.env
                print(f"\n--- Episode {ep+1} Step {step} ---")
                print(base.dump())

            time.sleep(0.1)

            if td["terminated"] or td["truncated"]:
                print(f"Episode {ep+1} finished after {step} steps, done={td['done']}")
                break
    env.close()


if __name__ == "__main__":
    train(total_timesteps=100000, n_envs=2)
    evaluate(episodes=1, is_dump=False)
