
from dataclasses import dataclass, field
from typing import Literal

import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  combine_frame_transforms,
  compute_pose_error,
  matrix_from_quat,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv


_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class UniformPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: "UniformPoseCommandCfg"
    """Configuration for the command generator."""

    def __init__(self, cfg: "UniformPoseCommandCfg", env: ManagerBasedRlEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Entity = env.scene[cfg.asset_name]
        self.body_indices = self.robot.find_bodies(cfg.body_names)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """
    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.pose_command_w[:, :3] + self._env.scene.env_origins[:, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.pose_command_w[:, 3:]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indices]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indices]

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_link_pos_w,
            self.robot.data.root_link_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_link_pos_w[:, self.body_indices],
            self.robot.data.body_link_quat_w[:, self.body_indices],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: torch.Tensor):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.pose_range["pos_x"])
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.pose_range["pos_y"])
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.pose_range["pos_z"])
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.pose_range["roll"])
        euler_angles[:, 1].uniform_(*self.cfg.pose_range["pitch"])
        euler_angles[:, 2].uniform_(*self.cfg.pose_range["yaw"])
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat

    def _update_command(self):
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        desired_body_pos = self.body_pos_w[visualizer.env_idx].cpu().numpy()
        desired_body_quat = self.body_quat_w[visualizer.env_idx]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[visualizer.env_idx].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[visualizer.env_idx]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        body_name = self.cfg.body_names
        visualizer.add_frame(
            position=desired_body_pos,
            rotation_matrix=desired_body_rotm,
            scale=0.08,
            label=f"desired_{body_name}",
            axis_colors=_DESIRED_FRAME_COLORS,
        )
        visualizer.add_frame(
            position=current_body_pos,
            rotation_matrix=current_body_rotm,
            scale=0.12,
            label=f"current_{body_name}",
        )


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type[UniformPoseCommand] = UniformPoseCommand

    asset_name: str
    body_names: tuple[str, ...]

    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)

    @dataclass
    class VizCfg:
        mode: Literal["ghost", "frames"] = "frames"
        ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)
