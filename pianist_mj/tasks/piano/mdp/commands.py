from dataclasses import dataclass, field
from typing import Literal
import math

import math
import torch
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.manager_term_config import CommandTermCfg, CommandTerm
from mjlab.managers import CommandTerm
from mjlab.viewer.debug_visualizer import DebugVisualizer

from pianist_mj.robots.piano_articulation import PianoArticulation

from pianist_mj.music.music_sequence import MusicSequence


FINGERTIP_COLORS = [
    # Important: the order of these colors should not be changed.
    (0.8, 0.2, 0.8),  # Purple.
    (0.8, 0.2, 0.2),  # Red.
    (0.2, 0.8, 0.8),  # Cyan.
    (0.2, 0.2, 0.8),  # Blue.
    (0.8, 0.8, 0.2),  # Yellow.

    (0.8, 0.2, 0.8),  # Purple.
    (0.8, 0.2, 0.2),  # Red.
    (0.2, 0.8, 0.8),  # Cyan.
    (0.2, 0.2, 0.8),  # Blue.
    (0.8, 0.8, 0.2),  # Yellow.
]

KEY_TARGET_COLOR = (0.2, 0.2, 0.2)

torch.set_printoptions(precision=2)


class KeyPressCommand(CommandTerm):
    """
    This command generates key press commands.
    """

    cfg: "KeyPressCommandCfg"
    """Configuration for the command generator."""

    def __init__(self, cfg: "KeyPressCommandCfg", env: ManagerBasedRlEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        assert self.cfg.song_name.endswith(".npz"), f"Invalid song name: {self.cfg.song_name}"
        self.song = MusicSequence.load(self.cfg.song_name, device=self.device)

        self.song.speedup(self.cfg.song_speedup)

        # extract the robot and body index for which the command is generated
        self.piano: PianoArticulation = env.scene[self.cfg.piano_name]
        # self.piano.post_scene_creation_init()  # cannot find a better place to call init

        if self.cfg.robot_name:
            self.robot: Entity = env.scene[self.cfg.robot_name]

            if self.cfg.right_hand_only:
                assert len(self.cfg.robot_finger_body_names) == 5
            else:
                assert len(self.cfg.robot_finger_body_names) == 10

            finger_body_indices, _ = self.robot.find_bodies(self.cfg.robot_finger_body_names, preserve_order=True)
            self._finger_body_indices = torch.tensor(finger_body_indices, device=self.device)

            self.num_fingers = len(self.cfg.robot_finger_body_names)
        else:
            self.num_fingers = 10

        # --- Create buffers ---
        # discrete command to indicate if the key needs to be pressed
        lookahead_slots = math.ceil(self.cfg.lookahead_steps / self.cfg.skip_stride)
        self._key_goal_states_lookahead = torch.zeros(self.num_envs, lookahead_slots, 88, dtype=torch.bool, device=self.device)
        # target locations of the keys to be pressed, maximum 10 keys
        self._fingertip_goal_locations = torch.zeros(self.num_envs, self.num_fingers, 3, device=self.device)

        # we only use this when using mocap config
        self._key_goal_locations = torch.zeros(self.num_envs, self.num_fingers, 3, device=self.device)
        self.desired_palm_poses = torch.tensor(self.cfg.desired_palm_pose, device=self.device)

        # use finger placement guidance
        if self.song.active_fingers.any():
            self.use_finger_placement_guidance = True
            print("Using finger placement guidance")
        else:
            self.use_finger_placement_guidance = False
            print("Dataset does not contain fingering annotations, not using finger placement guidance")

        if self.use_finger_placement_guidance:
            assert self.song.active_fingers.any()
            # discrete vector with 5 elements (thumb, index, middle, ring, pinky)
            self._active_fingers_lookahead = torch.zeros(self.num_envs, lookahead_slots, self.num_fingers, dtype=torch.bool, device=self.device)
        else:
            self._active_fingers_lookahead = torch.ones(self.num_envs, lookahead_slots, self.num_fingers, dtype=torch.bool, device=self.device)

        # step counter for the song
        self._song_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- metrics
        if self.cfg.robot_name:
            self.metrics["fingertip_to_goal_distance"] = torch.zeros(self.num_envs, device=self.device)

        # true positive rate (also known as recall) = TP / P
        self.metrics["true_positive_rate"] = torch.zeros(self.num_envs, device=self.device)
        # true negative rate = TN / N
        self.metrics["true_negative_rate"] = torch.zeros(self.num_envs, device=self.device)
        # precision = TP / (TP + FP)
        self.metrics["precision"] = torch.zeros(self.num_envs, device=self.device)
        # f1 score = 2 * TP / (2 * TP + FP + FN)
        self.metrics["f1"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "KeyPressCommand:\n"
        msg += f"\tSong name: {self.cfg.song_name}\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The command to the robot."""
        return self.key_goal_states_lookahead.flatten(start_dim=1).float()

    @property
    def key_goal_states(self) -> torch.Tensor:
        """The boolean goal key pressed states."""
        return self._key_goal_states_lookahead[:, 0, :]

    @property
    def key_goal_locations(self):
        """The goal key locations of the piano with scene origin offset."""
        return self._key_goal_locations

    @property
    def fingertip_goal_locations(self) -> torch.Tensor:
        """The goal fingertip locations of the robot with scene origin offset."""
        return self._fingertip_goal_locations

    @property
    def fingertip_locations(self) -> torch.Tensor:
        """The fingertip locations of the robot with scene origin offset."""
        return self.robot.data.body_link_pos_w[:, self._finger_body_indices]

    @property
    def active_fingers(self) -> torch.Tensor:
        return self._active_fingers_lookahead[:, 0, :]

    @property
    def key_goal_states_lookahead(self) -> torch.Tensor:
        return self._key_goal_states_lookahead

    @property
    def active_fingers_lookahead(self) -> torch.Tensor:
        assert self.use_finger_placement_guidance, "Dataset does not contain fingering annotations"
        return self._active_fingers_lookahead

    def _resample_command(self, env_ids: torch.Tensor):
        # restart the song at a random frame
        if self.cfg.random_restart:
            self._song_steps[env_ids] = torch.randint(0, self.song.num_frames, (env_ids.shape[0],), dtype=torch.int32, device=self.device)
        else:
            self._song_steps[env_ids] = 0  # for eval

        self._key_goal_states_lookahead[env_ids] = 0
        self._fingertip_goal_locations[env_ids] = 0.0
        self._key_goal_locations[env_ids] = 0.0

        if self.use_finger_placement_guidance:
            self._active_fingers_lookahead[env_ids] = 0

    def _update_command(self):
        self._song_steps[:] += 1
        env_ids = torch.where(self._song_steps >= self.song.num_frames)[0]
        self._resample_command(env_ids)

        lookahead_steps = self._song_steps.unsqueeze(1) + torch.arange(0, self.cfg.lookahead_steps, step=self.cfg.skip_stride, device=self.device)

        if self.use_finger_placement_guidance:
            active_keys_lookahead, active_fingers_lookahead, finger_placements_lookahead = self.song.get_frames(lookahead_steps)

            # for right-hand-only case, the right hand fingers should be the first 5 elements
            active_fingers_lookahead = active_fingers_lookahead[:, :, :self.num_fingers]
            finger_placements_lookahead = finger_placements_lookahead[:, :, :self.num_fingers]

            finger_placements_current = finger_placements_lookahead[:, 0]
            active_fingers_current = active_fingers_lookahead[:, 0]

            # create a selection tensor to select all environments
            all_env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(
                self.num_envs, self.num_fingers,
            )
            key_locations = self.piano.get_key_locations(all_env_ids, finger_placements_current)

            self._key_goal_states_lookahead[:] = active_keys_lookahead
            self._active_fingers_lookahead[:] = active_fingers_lookahead
            # mask the key locations with the active fingers
            self._fingertip_goal_locations[:] = key_locations * active_fingers_current.unsqueeze(-1)

        else:
            # HACK
            frame_indices = self.song._clamp_frame_indices(lookahead_steps)

            active_keys_lookahead = self.song.keys[frame_indices]
            active_keys_current = active_keys_lookahead[:, 0]
            active_key_indices = torch.nonzero(active_keys_current)
            env_ids = active_key_indices[:, 0]

            self._key_goal_states_lookahead[:] = active_keys_lookahead

            key_locations = self.piano.get_key_locations(env_ids, active_key_indices[:, 1])

            # Compare each element with all previous ones and create mask for "same env"
            same_env = env_ids.unsqueeze(1) == env_ids.unsqueeze(0)

            # Compute cumulative counts per group
            # For each i, count how many same keys appeared before
            env_cum_index = same_env.tril().sum(1) - 1

            # clear the key goal locations first
            self._key_goal_locations[:] = 0.0

            self._fingertip_goal_locations[:] = (
                self.song.fingertip_locations[frame_indices, :, :][:, 0, :]
                + torch.tensor([0.25, 0.62, 0.745], device=self.device)
                + self._env.scene.env_origins.unsqueeze(1)
            )
            self._key_goal_locations[env_ids, env_cum_index, :] = key_locations

    def get_goal_distances(self) -> torch.Tensor:
        """ the distance between goal and actual for all fingertips """
        return torch.norm(self.fingertip_goal_locations - self.fingertip_locations, dim=-1)

    def get_average_goal_distances(self) -> torch.Tensor:
        """ the average distance across all active fingers """
        num_active_fingers = self.active_fingers.sum(dim=-1).float()
        # only consider the distances for the active fingers
        effective_distances = self.active_fingers * self.get_goal_distances()
        return effective_distances.sum(dim=-1) / (num_active_fingers + 1e-6)

    def _update_metrics(self):
        if self.cfg.robot_name:
            distance_error = self.get_average_goal_distances()

        # TODO: note that when using single finger, the key press is still calculating for all 10 possible
        # keys, so the result metric will be lower than expected.

        # get the number of keys intended to be pressed and not pressed
        on_keys = self.key_goal_states
        off_keys = ~self.key_goal_states
        pressed_keys = self.piano.key_states
        not_pressed_keys = ~self.piano.key_states

        # Calculate precision and recall with zero_division=1 behavior (scikit-learn style)
        num_on_keys = on_keys.sum(dim=-1)
        num_off_keys = off_keys.sum(dim=-1)
        num_pressed = pressed_keys.sum(dim=-1).float()

        # compute the number of keys that are correctly pressed and not pressed
        correctly_pressed_count = (pressed_keys * on_keys).sum(dim=-1)
        correctly_not_pressed_count = (not_pressed_keys * off_keys).sum(dim=-1)

        # Precision = TP / (TP + FP), zero_division=1: return 1.0 when TP + FP = 0
        precision = torch.where(
            num_pressed > 0,
            correctly_pressed_count.float() / num_pressed,
            torch.ones_like(correctly_pressed_count.float())
        )

        # Recall = TP / (TP + FN), zero_division=1: return 1.0 when TP + FN = 0
        recall = torch.where(
            num_on_keys > 0,
            correctly_pressed_count.float() / num_on_keys.float(),
            torch.ones_like(correctly_pressed_count.float())
        )

        # F1 = 2 * (precision * recall) / (precision + recall), zero_division=1: return 1.0 when precision + recall = 0
        precision_plus_recall = precision + recall
        f1_score = torch.where(
            precision_plus_recall > 0,
            2 * precision * recall / precision_plus_recall,
            torch.ones_like(precision)
        )

        # Calculate true positive and negative rates
        true_positive_rate = recall  # recall is the same as true positive rate
        true_negative_rate = correctly_not_pressed_count.float() / (num_off_keys.float() + 1e-6)
        
        # if there are no keys intended to be not pressed, assign all correct to that environment
        # this is not possible in our case, so we skip this check:
        # true_negative_rate[num_off_keys == 0] = 1.0

        if self.cfg.robot_name:
            self.metrics["fingertip_to_goal_distance"] = distance_error
        self.metrics["true_positive_rate"] = true_positive_rate
        self.metrics["true_negative_rate"] = true_negative_rate
        self.metrics["precision"] = precision
        self.metrics["f1"] = f1_score

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        # create markers if necessary for the first time
        # if debug_vis:
        #     if not hasattr(self, "goal_pose_visualizers"):
        #         # -- goal pose
        #         self.goal_pose_visualizers = [
        #             VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
        #             for _ in range(1 if self.cfg.right_hand_only else 2)
        #         ]
        #     if not hasattr(self, "current_pose_visualizers"):
        #         # -- current pose
        #         self.current_pose_visualizers = [
        #             VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
        #             for _ in range(1 if self.cfg.right_hand_only else 2)
        #         ]
        #     if not hasattr(self, "goal_fingertip_visualizers"):
        #         # -- goal finger
        #         self.goal_fingertip_visualizers = []
        #         for i in range(5 if self.cfg.right_hand_only else 10):
        #             cfg = self.cfg.goal_key_visualizer_cfg.copy()
        #             cfg.markers["cuboid"].visual_material.diffuse_color = FINGERTIP_COLORS[i]
        #             self.goal_fingertip_visualizers.append(VisualizationMarkers(cfg))
        #     if not hasattr(self, "goal_key_visualizers"):
        #         # -- goal key
        #         self.goal_key_visualizers = []
        #         for i in range(5 if self.cfg.right_hand_only else 10):
        #             cfg = self.cfg.goal_key_visualizer_cfg.copy()
        #             cfg.markers["cuboid"].visual_material.diffuse_color = KEY_TARGET_COLOR
        #             self.goal_key_visualizers.append(VisualizationMarkers(cfg))
        #     if not hasattr(self, "current_key_visualizers"):
        #         # -- current key
        #         self.current_key_visualizers = []
        #         for i in range(5 if self.cfg.right_hand_only else 10):
        #             cfg = self.cfg.current_key_visualizer_cfg.copy()
        #             cfg.markers["sphere"].visual_material.diffuse_color = FINGERTIP_COLORS[i]
        #             self.current_key_visualizers.append(VisualizationMarkers(cfg))
        #     # set their visibility to true
        #     for goal_pose_visualizer in self.goal_pose_visualizers:
        #         goal_pose_visualizer.set_visibility(True)
        #     for current_pose_visualizer in self.current_pose_visualizers:
        #         current_pose_visualizer.set_visibility(True)

        #     for goal_fingertip_visualizer in self.goal_fingertip_visualizers:
        #         goal_fingertip_visualizer.set_visibility(True)
        #     for goal_key_visualizer in self.goal_key_visualizers:
        #         goal_key_visualizer.set_visibility(True)
        #     for current_key_visualizer in self.current_key_visualizers:
        #         current_key_visualizer.set_visibility(True)

        # else:
        #     if hasattr(self, "goal_pose_visualizers"):
        #         for goal_pose_visualizer in self.goal_pose_visualizers:
        #             goal_pose_visualizer.set_visibility(False)
        #     if hasattr(self, "current_pose_visualizers"):
        #         for current_pose_visualizer in self.current_pose_visualizers:
        #             current_pose_visualizer.set_visibility(False)
        #     if hasattr(self, "goal_fingertip_visualizers"):
        #         for goal_fingertip_visualizer in self.goal_fingertip_visualizers:
        #             goal_fingertip_visualizer.set_visibility(False)
        #     if hasattr(self, "goal_key_visualizers"):
        #         for goal_key_visualizer in self.goal_key_visualizers:
        #             goal_key_visualizer.set_visibility(False)
        #     if hasattr(self, "current_key_visualizers"):
        #         for current_key_visualizer in self.current_key_visualizers:
        #             current_key_visualizer.set_visibility(False)
        pass


@dataclass(kw_only=True)
class KeyPressCommandCfg(CommandTermCfg):
    """Configuration for key press command generator."""

    class_type: type = KeyPressCommand

    resampling_time_range: tuple[float, float] = field(default_factory=lambda: (1e9, 1e9))

    song_name: str
    """Name of the song in the environment for which the commands are generated."""

    piano_name: str
    """Name of the piano in the environment for which the commands are generated."""

    robot_name: str
    """Name of the robot in the environment for which the commands are generated."""

    robot_finger_body_names: list[str] = field(default_factory=list)
    """Names of the robot finger bodies for which the commands are generated."""

    robot_palm_body_names: list[str] = field(default_factory=list)
    """Names of the robot palm bodies for which the commands are generated."""

    desired_palm_pose: list[tuple[float, float, float, float]] = field(default_factory=list)
    """The desired palm pose for the robot."""

    song_speedup: float
    """The factor to speed up the tempo of the song."""

    lookahead_steps: int
    """The number of steps to look ahead."""

    skip_stride: int
    """The stride to skip the lookahead steps."""

    right_hand_only: bool
    """Whether to only use the right hand for the commands."""

    random_restart: bool
    """Whether to restart the song at a random frame."""

    # goal_pose_visualizer_cfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/goal_pose",
    #     markers={
    #         "frame": sim_utils.UsdFileCfg(
    #             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
    #             scale=(0.05, 0.05, 0.05),
    #         ),
    #         "connecting_line": sim_utils.CylinderCfg(
    #             radius=0.002,
    #             height=1.0,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), roughness=1.0),
    #         ),
    #     }
    # )
    # """The configuration for the goal pose visualization marker."""

    # current_pose_visualizer_cfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/current_pose",
    #     markers={
    #         "frame": sim_utils.UsdFileCfg(
    #             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
    #             scale=(0.05, 0.05, 0.05),
    #         ),
    #         "connecting_line": sim_utils.CylinderCfg(
    #             radius=0.002,
    #             height=1.0,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), roughness=1.0),
    #         ),
    #     }
    # )
    # """The configuration for the current pose visualization marker."""

    # goal_key_visualizer_cfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/goal_key",
    #     markers={
    #         "cuboid": sim_utils.CuboidCfg(
    #             size=(0.05, 0.02, 0.025),
    #             visual_material=sim_utils.PreviewSurfaceCfg(),
    #         ),
    #     },
    # )
    # """The configuration for the goal pose visualization marker."""

    # current_key_visualizer_cfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/current_key",
    #     markers={
    #         "sphere": sim_utils.SphereCfg(
    #             radius=0.01,
    #             visual_material=sim_utils.PreviewSurfaceCfg(),
    #         ),
    #     },
    # )
    # """The configuration for the current pose visualization marker."""
