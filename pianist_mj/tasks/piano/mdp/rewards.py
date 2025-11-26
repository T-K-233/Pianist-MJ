import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.sensor.contact_sensor import ContactSensor as ContactSensor

from pianist_mj.tasks.piano.mdp.commands import KeyPressCommand
from pianist_mj.tasks.piano.mdp.math_functions import windowed_gaussian
from pianist_mj.robots.piano_articulation import PianoArticulation


# each reward term should return a tensor of shape (num_envs,)

def key_on_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    piano_cfg: SceneEntityCfg = SceneEntityCfg("piano"),
) -> torch.Tensor:
    """Reward for pressing the right keys at the right time."""
    command_term = env.command_manager.get_term(command_name)
    assert isinstance(command_term, KeyPressCommand)

    piano: PianoArticulation = env.scene[piano_cfg.name]
    correct_keys = torch.logical_and(command_term.key_goal_states, piano.key_states)
    return correct_keys.sum(dim=-1).float()


def key_off_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    piano_cfg: SceneEntityCfg = SceneEntityCfg("piano"),
) -> torch.Tensor:
    """Reward for not pressing the wrong keys."""
    command_term = env.command_manager.get_term(command_name)
    assert isinstance(command_term, KeyPressCommand)

    piano: PianoArticulation = env.scene[piano_cfg.name]

    false_positives = torch.logical_and(command_term.key_goal_states.logical_not(), piano.key_states)

    return false_positives.any(dim=-1).logical_not().float()


def key_position_error_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
    piano_cfg: SceneEntityCfg = SceneEntityCfg("piano"),
) -> torch.Tensor:
    """Compute the L1 error between the goal and actual key positions."""
    command_term = env.command_manager.get_term(command_name)
    assert isinstance(command_term, KeyPressCommand)

    piano: PianoArticulation = env.scene[piano_cfg.name]

    errors = torch.abs(command_term.key_goal_states.float() - piano.key_positions)
    return errors.sum(dim=-1)


def joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def joint_energy_l1(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint energy using L1 kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    powers = torch.abs(asset.data.actuator_force * asset.data.joint_vel)
    return torch.sum(powers, dim=-1)


def joint_deviation_l1(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def fingertip_to_goal_distance_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    distance_threshold: float = 0.01,
    std: float = 0.05,
) -> torch.Tensor:
    """Reward for minimizing the distance between the fingertip and the goal."""
    command_term = env.command_manager.get_term(command_name)
    assert isinstance(command_term, KeyPressCommand)

    # get the distances between all of the fingertips and the goal
    all_distances = command_term.get_goal_distances()
    all_distance_rewards = windowed_gaussian(
        all_distances,
        lower=0,
        upper=distance_threshold,
        std=std,
    )
    # mask off the rewards for the inactive fingers
    active_rewards = command_term.active_fingers * all_distance_rewards
    rewards = active_rewards.sum(dim=-1)

    return rewards


def key_clearance(
    env: ManagerBasedRlEnv,
    command_name: str,
    z_threshold: float = 0.8,
    std: float = 0.02,
) -> torch.Tensor:
    """Reward for keeping a Z-axis clearance between the key and the fingertip when not pressing."""
    command_term = env.command_manager.get_term(command_name)
    assert isinstance(command_term, KeyPressCommand)
    z_distance_error = torch.clamp(z_threshold - command_term.fingertip_locations[:, :, 2], min=0.0).mean(dim=-1)
    error = z_distance_error * command_term.active_fingers.any(dim=-1).logical_not()
    return torch.exp(-(error**2) / std**2)


def palm_orientation_l2(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize hand palm orientation using L2 squared kernel.

    This is computed by penalizing the yz-components of the body pose.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    assert isinstance(command_term, KeyPressCommand)
    error = quat_error_magnitude(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids],
        command_term.desired_palm_poses.repeat(env.num_envs, 1, 1).to(asset.device),
    )
    return torch.exp(-0.5 * (error**2).mean(-1) / std**2)


# === RobotPianist Rewards === #

def robopianist_key_press_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    piano_cfg: SceneEntityCfg = SceneEntityCfg("piano"),
) -> torch.Tensor:
    """Reward for pressing the right keys at the right time."""
    command_term = env.command_manager.get_term(command_name)
    assert isinstance(command_term, KeyPressCommand)
    piano: PianoArticulation = env.scene[piano_cfg.name]

    # on = np.flatnonzero(self._goal_current[:-1])
    on = command_term.key_goal_states

    rew = 0.0
    # It's possible we have no keys to press at this timestep, so we need to check
    # that `on` is not empty.
    if on.any():
        actual = piano.key_positions

        rews = windowed_gaussian(
            command_term.key_goal_states.float() - actual,
            lower=0,
            upper=0.01,
            std=0.01,
        )
        # rews = tolerance(
        #     self._goal_current[:-1][on] - actual[on],
        #     bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
        #     margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
        #     sigmoid="gaussian",
        # )
        rew += 0.5 * rews.mean()

    # If there are any false positives, the remaining 0.5 reward is lost.
    # off = np.flatnonzero(1 - self._goal_current[:-1])
    # rew += 0.5 * (1 - float(self.piano.activation[off].any()))
    false_positives = torch.logical_and(command_term.key_goal_states.logical_not(), piano.key_states)
    rew += 0.5 * false_positives.any(dim=-1).logical_not()

    breakpoint()

    return rew


def robopianist_energy_reward(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Reward for minimizing energy."""
    # rew = 0.0
    # for hand in [self.right_hand, self.left_hand]:
    #     power = hand.observables.actuators_power(physics).copy()
    #     rew -= self._energy_penalty_coef * np.sum(power)
    rew = -joint_energy_l1(env, asset_cfg=SceneEntityCfg("robot", joint_names=[".*"])).sum(dim=-1)
    return rew


def robopianist_fingering_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward for minimizing the distance between the fingers and the keys."""
    command_term = env.command_manager.get_term(command_name)
    assert isinstance(command_term, KeyPressCommand)

    # def _distance_finger_to_key(
    #     hand_keys: List[Tuple[int, int]], hand
    # ) -> List[float]:
    #     distances = []
    #     for key, mjcf_fingering in hand_keys:
    #         fingertip_site = hand.fingertip_sites[mjcf_fingering]
    #         fingertip_pos = physics.bind(fingertip_site).xpos.copy()
    #         key_geom = self.piano.keys[key].geom[0]
    #         key_geom_pos = physics.bind(key_geom).xpos.copy()
    #         key_geom_pos[-1] += 0.5 * physics.bind(key_geom).size[2]
    #         key_geom_pos[0] += 0.35 * physics.bind(key_geom).size[0]
    #         diff = key_geom_pos - fingertip_pos
    #         distances.append(float(np.linalg.norm(diff)))
    #     return distances

    distances = command_term.get_goal_distances()

    # # Case where there are no keys to press at this timestep.
    # if not distances:
    #     return 0.0
    distances *= command_term.active_fingers

    # rews = tolerance(
    #     np.hstack(distances),
    #     bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
    #     margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
    #     sigmoid="gaussian",
    # )
    rews = windowed_gaussian(distances, lower=0, upper=0.01, std=0.01)

    breakpoint()
    return rews.sum(dim=-1)


def undesired_contacts(env: ManagerBasedRlEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: Entity = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    zero_contact = (~contacts).all(dim=1)
    return 1.0 * zero_contact


def contact_forces(env: ManagerBasedRlEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)
