"""Prime constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
    ElectricActuator,
    reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PRIME_XML: Path = (
    Path("data") / "assets" / "robots" / "prime" / "mjcf" / "prime_upper_body.xml"
)
assert PRIME_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, PRIME_XML.parent.parent / "meshes", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PRIME_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


EROB70F_50_PARAMS = {
    "effort_limit": 12.0,  # Nm
    "velocity_limit": 6.2832,  # rad/s
    "armature": 0.1172,  # kgm^2
}
""" Parameters for the eRob70F 50 reduction ratio actuator. """

EROB70HV5_50_PARAMS = {
    "effort_limit": 23.0,  # Nm
    "velocity_limit": 6.2832,  # rad/s
    "armature": 0.0653,  # kgm^2
}
""" Parameters for the eRob70HV5 50 reduction ratio actuator. """

EROB70HV5_80_PARAMS = {
    "effort_limit": 30.0,  # Nm
    "velocity_limit": 3.9270,  # rad/s
    "armature": 0.1672,  # kgm^2
}
""" Parameters for the eRob70HV5 80 reduction ratio actuator. """

EROB70HV5_120_PARAMS = {
    "effort_limit": 36.0,  # Nm
    "velocity_limit": 2.6180,  # rad/s
    "armature": 0.3762,  # kgm^2
}
""" Parameters for the eRob70HV5 120 reduction ratio actuator. """

EROB80HV6_50_PARAMS = {
    "effort_limit": 44.0,  # Nm
    "velocity_limit": 6.2832,  # rad/s
    "armature": 0.1124,  # kgm^2
}
""" Parameters for the eRob80HV6 50 reduction ratio actuator. """

EROB80HV6_80_PARAMS = {
    "effort_limit": 56.0,  # Nm
    "velocity_limit": 3.9270,  # rad/s
    "armature": 0.2878,  # kgm^2
}
""" Parameters for the eRob80HV6 80 reduction ratio actuator. """

EROB80HV6_120_PARAMS = {
    "effort_limit": 70.0,  # Nm
    "velocity_limit": 2.6180,  # rad/s
    "armature": 0.6474,  # kgm^2
}
""" Parameters for the eRob80HV6 120 reduction ratio actuator. """

TA40_50_PARAMS = {
    "effort_limit": 6.6,  # Nm
    "velocity_limit": 8.3776,  # rad/s
    "armature": 0.0375,  # kgm^2
}
""" Parameters for the TA-40 50 reduction ratio actuator. """

TA40_50_PARAMS = {
    "effort_limit": 3.3,  # Nm
    "velocity_limit": 8.3776,  # rad/s
    "armature": 0.0375,  # kgm^2
}
""" Parameters for the TA-40 100 reduction ratio actuator. """

TA40_100_PARAMS = {
    "effort_limit": 4.8,  # Nm
    "velocity_limit": 4.1888,  # rad/s
    "armature": 0.1500,  # kgm^2
}
""" Parameters for the TA-40 100 reduction ratio actuator. """


##
# Actuator config.
##

EROB80H_120_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*_shoulder_pitch", ".*_shoulder_roll"),
    stiffness=230.04,
    damping=48.82,
    effort_limit=EROB80HV6_120_PARAMS["effort_limit"],
    armature=EROB80HV6_120_PARAMS["armature"],
)
EROB80H_80_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*_shoulder_yaw", ),
    stiffness=284.00,
    damping=36.16,
    effort_limit=EROB80HV6_80_PARAMS["effort_limit"],
    armature=EROB80HV6_80_PARAMS["armature"],
)
EROB70H_120_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*_elbow_pitch", ),
    stiffness=371.34,
    damping=47.28,
    effort_limit=EROB70HV5_120_PARAMS["effort_limit"],
    armature=EROB70HV5_120_PARAMS["armature"],
)
EROB70H_80_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*_wrist_yaw", ),
    stiffness=165.04,
    damping=21.01,
    effort_limit=EROB70HV5_80_PARAMS["effort_limit"],
    armature=EROB70HV5_80_PARAMS["armature"],
)
TA40H_100_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*_wrist_roll", ".*_wrist_pitch"),
    stiffness=148.0,
    damping=18.85,
    effort_limit=TA40_100_PARAMS["effort_limit"],
    armature=TA40_100_PARAMS["armature"],
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(-0.35, 0, 0.7),
    joint_pos={
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},
)

HAND_RAISED_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(-0.35, 0, 0.7),
    joint_pos={
        "left_shoulder_pitch": -0.350,
        "left_shoulder_roll": 0.3,
        "left_shoulder_yaw": 0.0,
        "left_elbow_pitch": -1.5708,
        "left_wrist_yaw": -1.2708,
        "left_wrist_roll": 0.0,
        "left_wrist_pitch": 0.2,
        "right_shoulder_pitch": -0.350,
        "right_shoulder_roll": -0.3,
        "right_shoulder_yaw": 0.0,
        "right_elbow_pitch": -1.5708,
        "right_wrist_yaw": 1.2708,
        "right_wrist_roll": 0.0,
        "right_wrist_pitch": -0.2,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot[1-7]_collision$": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=0,
    conaffinity=1,
    condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot[1-7]_collision$": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)

##
# Final config.
##

PRIME_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        EROB80H_120_CFG,
        EROB80H_80_CFG,
        EROB70H_120_CFG,
        EROB70H_80_CFG,
        TA40H_100_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_prime_robot_cfg() -> EntityCfg:
    """Get a fresh Prime robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=PRIME_ARTICULATION,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_prime_robot_cfg())

    viewer.launch(robot.spec.compile())
