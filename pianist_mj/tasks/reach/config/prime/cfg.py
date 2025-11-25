"""Unitree G1 velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from pianist_mj.tasks.reach.cfg import make_env_cfg
from pianist_mj.robots.prime.prime_constants import get_prime_robot_cfg


def prime_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Prime reach environment configuration."""
    cfg = make_env_cfg()

    cfg.scene.entities["robot"] = get_prime_robot_cfg()

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="body_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="body_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (self_collision_cfg, )

    cfg.viewer.body_name = "body_base"

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def prime_reach_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Unitree G1 velocity task."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=2000,
        save_interval=200,
        experiment_name="reach_prime",
        run_name="",
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=(128, 64, 64),
            critic_hidden_dims=(128, 64, 64),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0005,
            num_learning_epochs=8,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
