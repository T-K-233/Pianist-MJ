"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import numpy as np
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

from pianist_mj.tasks.reach import mdp


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
        "left_pose": mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_names=("left_wrist_pitch",),
            resampling_time_range=(1.0, 4.0),
            pose_range={
                "pos_x": (0.35, 0.65),
                "pos_y": (-0.2, 0.6),
                "pos_z": (0.15, 0.5),
                "roll": (
                    (0.5 * np.pi) - (0.25 * np.pi),
                    (0.5 * np.pi) + (0.25 * np.pi),
                ),
                "pitch": (
                    -(0.5 * np.pi) - (0.25 * np.pi),
                    -(0.5 * np.pi) + (0.25 * np.pi),
                ),
                "yaw": (
                    -(0.25 * np.pi),
                    (0.25 * np.pi),
                ),
            },
            debug_vis=True,
        ),
        "right_pose": mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_names=("right_wrist_pitch",),
            resampling_time_range=(1.0, 4.0),
            pose_range={
                "pos_x": (0.35, 0.65),
                "pos_y": (-0.6, 0.2),
                "pos_z": (0.15, 0.5),
                "roll": (
                    -(0.5 * np.pi) - (0.25 * np.pi),
                    -(0.5 * np.pi) + (0.25 * np.pi),
                ),
                "pitch": (
                    -(0.5 * np.pi) - (0.25 * np.pi),
                    -(0.5 * np.pi) + (0.25 * np.pi),
                ),
                "yaw": (
                    -(0.25 * np.pi),
                    (0.25 * np.pi),
                ),
            },
            debug_vis=True,
        ),
    }

    ##
    # Observations
    ##

    policy_terms = {
        "left_pose_command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "left_pose"},
        ),
        "right_pose_command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "right_pose"},
        ),
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
            scale=0.5,
            preserve_order=True,
            use_default_offset=True,
        )
    }

    ##
    # Rewards
    ##

    rewards = {
        "left_position_tracking": RewardTermCfg(
            func=mdp.position_command_error,
            params={
                "command_name": "left_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_pitch"),
            },
            weight=-1.0,
        ),
        "right_position_tracking": RewardTermCfg(
            func=mdp.position_command_error,
            params={
                "command_name": "right_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_pitch"),
            },
            weight=-1.0,
        ),
        "action_rate": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.05,
        ),
        "joint_vel": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-1e-4,
        ),
        "joint_energy": RewardTermCfg(
            func=mdp.joint_energy_l1,
            weight=-1e-4,
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-1.0,
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
            num_envs=1,
            extent=2.0,
            entities={
                "piano": EntityCfg(
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
