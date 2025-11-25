# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import combine_frame_transforms

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def joint_energy_l1(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize joint energy using L1 kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    powers = torch.abs(asset.data.actuator_force * asset.data.joint_vel)
    return torch.sum(powers, dim=-1)


def position_command_error(
    env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]

    des_pos_w, _ = combine_frame_transforms(asset.data.root_link_pos_w, asset.data.root_link_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0]]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


# def position_command_error_tanh(
#     env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
# ) -> torch.Tensor:
#     """Reward tracking of the position using the tanh kernel.

#     The function computes the position error between the desired position (from the command) and the
#     current position of the asset's body (in world frame) and maps it with a tanh kernel.
#     """
#     # extract the asset (to enable type hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # obtain the desired and current positions
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
#     curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
#     distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
#     return 1 - torch.tanh(distance / std)


# def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize tracking orientation error using shortest path.

#     The function computes the orientation error between the desired orientation (from the command) and the
#     current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
#     path between the desired and current orientations.
#     """
#     # extract the asset (to enable type hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # obtain the desired and current orientations
#     des_quat_b = command[:, 3:7]
#     des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
#     curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
#     return quat_error_magnitude(curr_quat_w, des_quat_w)
