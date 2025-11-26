"""Unitree G1 velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from pianist_mj.robots.prime.prime_wuji_constants import get_prime_wuji_robot_cfg
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    ActionTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.utils.os import update_assets

from pianist_mj.robots.prime.prime_wuji_constants import PRIME_WUJI_FINGER_BODY_NAMES, PRIME_WUJI_PALM_BODY_NAMES
from pianist_mj.robots.piano_articulation import PianoArticulationCfg

from pianist_mj.tasks.piano import mdp  # noqa: F401, F403


PIANO_XML: Path = (
    Path("data") / "assets" / "piano" / "mjcf" / "piano.xml"
)
assert PIANO_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, PIANO_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PIANO_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


def make_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base velocity tracking task configuration."""

    ##
    # Commands
    ##

    commands: dict[str, CommandTermCfg] = {
        "keypress": mdp.KeyPressCommandCfg(
            # song_name="./source/pianist/data/music/pig_single_finger/nocturne_op9_no_2-1.npz",
            song_name="data/music/etude/twinkle_twinkle_rousseau.npz",
            piano_name="piano",
            robot_name="robot",
            robot_finger_body_names=PRIME_WUJI_FINGER_BODY_NAMES,
            robot_palm_body_names=PRIME_WUJI_PALM_BODY_NAMES,
            desired_palm_pose=[(0.707, 0, 0.707, 0), (0.707, 0, 0.707, 0)],
            song_speedup=1,
            right_hand_only=False,
            random_restart=True,
            lookahead_steps=12,
            skip_stride=1,
            debug_vis=True,
        )
    }

    ##
    # Observations
    ##

    policy_terms = {
        # "left_pose_command": ObservationTermCfg(
        #     func=mdp.generated_commands,
        #     params={"command_name": "left_pose"},
        # ),
        # "right_pose_command": ObservationTermCfg(
        #     func=mdp.generated_commands,
        # params={"command_name": "right_pose"},
        # ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    critic_terms = {
        **policy_terms,
    }

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            scale=0.25,
            preserve_order=True,
            use_default_offset=True,
        )
    }

    ##
    # Rewards
    ##

    rewards = {
        "key_on": RewardTermCfg(
            func=mdp.key_on_reward,
            params={
                "command_name": "keypress",
                "piano_cfg": SceneEntityCfg("piano"),
            },
            weight=0.1,
        ),
        "key_position_error": RewardTermCfg(
            func=mdp.key_position_error_l1,
            params={
                "command_name": "keypress",
                "piano_cfg": SceneEntityCfg("piano"),
            },
            weight=-1.0,
        ),
        "minimize_fingertip_to_goal_distance_coarse": RewardTermCfg(
            func=mdp.fingertip_to_goal_distance_reward,
            params={
                "command_name": "keypress",
                "distance_threshold": 0.01,
                "std": 0.1,
            },
            weight=0.5,
        ),
        "minimize_fingertip_to_goal_distance_fine": RewardTermCfg(
            func=mdp.fingertip_to_goal_distance_reward,
            params={
                "command_name": "keypress",
                "distance_threshold": 0.01,
                "std": 0.02,
            },
            weight=1.0,
        ),
        "action_rate": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-2e-4,
        ),
        "joint_energy": RewardTermCfg(
            func=mdp.joint_energy_l1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            weight=-5.0e-5,
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            weight=-1.0,
        ),
        "joint_deviation_fingers": RewardTermCfg(
            func=mdp.joint_deviation_l1,
            params={        
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[  # excluding joint 2 to encourage lifting
                        "(left|right)_finger1_joint(3|4)",
                        "(left|right)_finger(2|3|4|5)_joint(1|3|4)",
                    ],
                )
            },
            weight=-0.05,
        ),
        "joint_deviation_arm": RewardTermCfg(
            func=mdp.joint_deviation_l1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["(left|right)_(shoulder|elbow|wrist)_.*"],
                )
            },
            weight=-5e-3,
        ),
        # "undesired_contacts": RewardTermCfg(
        #     func=mdp.undesired_contacts,
        #     params={
        #         "threshold": 1.0,
        #         "sensor_cfg": SceneEntityCfg(
        #             "contact_forces",
        #             body_names=["(left|right)_finger(1|2|3|4|5)_link(1|2|3)"],
        #         ),
        #     },
        #     weight=-0.5,
        # ),
        # "contact_forces": RewardTermCfg(
        #     func=mdp.contact_forces,
        #     params={
        #         "threshold": 50.0,
        #         "sensor_cfg": SceneEntityCfg(
        #             "contact_forces",
        #             body_names=["(left|right)_finger(1|2|3|4|5)_tip_link"],
        #         ),
        #     },
        #     weight=-1e-4,
        # ),
        "palm_orientation": RewardTermCfg(
            func=mdp.palm_orientation_l2,
            params={
                "command_name": "keypress",
                "asset_cfg": SceneEntityCfg("robot", body_names=["(left|right)_wrist_pitch"]),
                "std": 0.3,
            },
            weight=0.2,
        ),
    }

    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }

    ##
    # Events
    ##

    events = {
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.5, 1.5),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    ##
    # Curriculum
    ##

    curriculum = {}

    ##
    # Assemble and return
    ##

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(),
            num_envs=4096,
            extent=2.0,
            entities={
                "piano": PianoArticulationCfg(
                    init_state=EntityCfg.InitialStateCfg(
                        pos=(0.25, 0.0, 0.61),
                        rot=(0.0, 0.0, 0.0, 1.0),
                    ),
                    spec_fn=get_spec,
                )
            },
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            asset_name="robot",
            body_name="",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )


def piano_prime_wuji_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Prime reach environment configuration."""
    cfg = make_env_cfg()

    cfg.scene.entities["robot"] = get_prime_wuji_robot_cfg()

    self_collision_cfg = ContactSensorCfg(
        name="contact_forces",
        primary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (self_collision_cfg, )

    cfg.viewer.body_name = "base"

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


def piano_prime_wuji_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Unitree G1 velocity task."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=2000,
        save_interval=200,
        experiment_name="piano_prime_wuji",
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
