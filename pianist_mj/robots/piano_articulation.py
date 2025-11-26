import torch

from mjlab.envs import ManagerBasedRlEnv


NUM_KEYS = 88
""" 88 keys on a piano. """

PIANO_JOINTS = [f"key_{i}_joint" for i in range(NUM_KEYS)]


WHITE_KEY_INDICES = [
    0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26, 27, 29, 31, 32, 34, 36, 38, 39, 41, 43, 44, 46, 48, 50, 51, 53, 55, 56, 58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 86, 87,
]
""" Indices of the white keys. """

BLACK_TWIN_KEY_INDICES = [
    4, 6, 16, 18, 28, 30, 40, 42, 52, 54, 64, 66, 76, 78
]
""" Indices of the black twin keys. """

BLACK_TRIPLET_KEY_INDICES = [
    1, 9, 11, 13, 21, 23, 25, 33, 35, 37, 45, 47, 49, 57, 59, 61, 69, 71, 73, 81, 83, 85
]
""" Indices of the black triplet keys. """

BLACK_KEY_INDICES = BLACK_TWIN_KEY_INDICES + BLACK_TRIPLET_KEY_INDICES
""" Indices of the black keys. """

WHITE_KEY_LENGTH = 0.15  # 150 mm
""" Length of the visible portion of the white key. """

WHITE_KEY_TOTAL_LENGTH = 0.19  # 190 mm
""" Total length of the white key, this is distance between the tip and the rotation axis. """

# Black key dimensions
BLACK_KEY_WIDTH = 0.01  # 10 mm
""" Width at the top of the black key. """

BLACK_KEY_LENGTH = 0.09  # 90 mm
""" Length of the visible portion of the black key. """

BLACK_KEY_TOTAL_LENGTH = 0.12  # 120 mm
""" Total length of the black key, this is distance between the tip and the rotation axis. """

# Unlike the other dimensions, the height of the black key was roughly set such that
# when a white key is fully depressed, the bottom of the black key is barely visible.
BLACK_KEY_HEIGHT = 0.018
""" Height of the black key. """

BLACK_OFFSET_FROM_WHITE = 0.0125  # 12.5 mm
""" Offset from the top of the white key to the top of the black key. """

# Joint spring reference, in rad
# setting the zero position to be upwards to create
# a restoring force when the key is at rest
KEY_SPRINGREF = -0.01


class PianoArticulation():
    def __init__(self, env: ManagerBasedRlEnv):
        super().__init__()

        self.env = env

        self._initialized = False

        # we need to do these initializations manually after the scene is created.
        self.key_names = []

        for index, key_name in enumerate(PIANO_JOINTS):
            if index in WHITE_KEY_INDICES:
                self.key_names.append(f"white_{key_name}")
            else:
                self.key_names.append(f"black_{key_name}")
        key_body_indices, _ = self.env.scene["piano"].find_bodies([key_name.replace("_joint", "") for key_name in self.key_names], preserve_order=True)
        key_joint_indices, _ = self.env.scene["piano"].find_joints([key_name.replace("_joint", "").replace("_key", "_joint") for key_name in self.key_names], preserve_order=True)

        # ensure these are 1D tensors
        self._key_body_indices = torch.tensor(key_body_indices, device=self.env.device)
        self._key_joint_indices = torch.tensor(key_joint_indices, device=self.env.device)

        self._key_contact_offsets = torch.zeros(self.env.scene["piano"].num_joints, 3, device=self.env.device)

        # specify the contact position to be at 85% of the key length
        self._key_contact_offsets[WHITE_KEY_INDICES, 0] += -0.85 * WHITE_KEY_TOTAL_LENGTH
        self._key_contact_offsets[BLACK_KEY_INDICES, 0] += -0.85 * BLACK_KEY_TOTAL_LENGTH
        self._key_contact_offsets[BLACK_KEY_INDICES, 2] += 0.2 * BLACK_KEY_HEIGHT

        # gives an upward force when the key is at rest
        self.num_joints = 88
        self.device = self.env.device

        spring_ref_position = torch.zeros(1, self.num_joints, device=self.device)
        spring_ref_position[:] = KEY_SPRINGREF
        # TODO: fix it
        # self.set_joint_position_target(spring_ref_position)

        # self.synth = Synthesizer()
        # self.synth.start()
        # self.prev_pressed_keys = []
        # self.midi_offset = 21
        self._initialized = True

    @property
    def key_positions(self) -> torch.Tensor:
        """The normalized key joint positions."""
        key_position_normalized = self.env.scene["piano"].data.joint_pos / self.env.scene["piano"].data.default_joint_pos_limits[:, :, 1]
        key_position_normalized = torch.where(key_position_normalized < 1e-6, 0.0, key_position_normalized)
        return key_position_normalized[:, self._key_joint_indices]

    @property
    def key_velocities(self) -> torch.Tensor:
        """The normalized key joint velocities."""
        max_key_velocity = 10.0  # TODO: this is a bit arbitrary, but it works for now
        key_velocity_normalized = torch.clamp(
            self.env.scene["piano"].data.joint_vel / max_key_velocity,
            min=-1.0, max=1.0,
        )
        key_velocity_normalized = torch.where(key_velocity_normalized < 1e-6, 0.0, key_velocity_normalized)
        return key_velocity_normalized[:, self._key_joint_indices]

    @property
    def key_states(self) -> torch.Tensor:
        """The boolean key pressed states."""
        # TODO: fix hardcoded threshold
        return self.key_positions > 0.5

    def get_key_locations(self, env_ids: torch.Tensor, key_indices: torch.Tensor) -> torch.Tensor:
        """Get the world location of the key contacts with scene origin offset."""
        key_locations = self.env.scene["piano"].data.body_link_pos_w[env_ids, self._key_body_indices[key_indices]]
        # add the offset from the key rotate joints as the desired contact location
        key_locations += self._key_contact_offsets[key_indices]

        return key_locations

    def update(self, dt: float):
        super().update(dt)

        if not self._initialized:
            return
