import torch
import numpy as np


class MusicSequence:
    """
    The MusicSequence stores the sheet music and additional guidance information to play a piano
    music piece.

    Depending on the type, it contains the following data:

    All:
    - keys: A boolean mask that informs which of the 88 keys should be pressed. The keys are
      ordered by the piano key ordering from left to right (A0 to C8).
    - finger_names: The names of the fingers. Ordering is right hand thumb to pinky, then left
      hand thumb to pinky.

    Finger Placements:
    - active_fingers: A boolean mask that informs which of the 10 fingers are active (pressing a
      key).
    - finger_placements: A (10 -> 88) mapping of each finger needs to press which key.

    Mocap:
    - keypoint_names: The names of the mocap keypoints. Ordering follows the SMPL hand keypoint
      ordering.
    - joint_names: The names of the mocap joints. Ordering follows the SMPL hand joint ordering.
    - fingertip_locations: Reference motion fingertip positions in world Cartesian frame.
    - joint_positions: Reference motion joint positions.
    - keypoint_locations: Reference motion keypoint positions in world Cartesian frame.
    """
    def __init__(
        self,
        num_frames: int,
        dt: float,
        num_keys: int = 88,
        finger_names: list[str] = [],
        joint_names: list[str] = [],
        keypoint_names: list[str] = [],
        device: torch.device | str = torch.device("cpu"),
    ):
        self.num_frames = num_frames
        self.dt = dt  # sample rate in seconds
        self.duration = num_frames * dt  # duration in seconds
        self._device = device
        self.num_keys = num_keys

        # stores boolean masks of keys that should be pressed at each frame
        self.keys = torch.zeros(self.num_frames, num_keys, dtype=torch.bool, device=device)
        # finger names, should be in order of
        # [right thumb, right index, right middle, right ring, right pinky, left thumb, left index, left middle, left ring, left pinky]
        self.finger_names = finger_names
        self.num_fingers = len(finger_names)

        # Finger Placements-based data
        # stores boolean masks of fingers that are active at each frame
        self.active_fingers = torch.zeros(self.num_frames, self.num_fingers, dtype=torch.bool, device=device)
        # stores the finger placements (which finger should press which key, a 10->88 mapping)
        self.finger_placements = torch.zeros(self.num_frames, self.num_fingers, dtype=torch.int32, device=device)

        # Mocap-based data
        self.joint_names = joint_names
        self.num_joints = len(joint_names)
        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)
        # stores the reference motion fingertip Cartesian positions
        self.fingertip_locations = torch.zeros(self.num_frames, self.num_fingers, 3, dtype=torch.float32, device=device)
        # stores the reference motion joint positions of the robot hand and arm
        self.joint_positions = torch.zeros(self.num_frames, self.num_joints, dtype=torch.float32, device=device)
        # stores the reference motion keypoint Cartesian positions
        self.keypoint_locations = torch.zeros(self.num_frames, self.num_keypoints, 3, dtype=torch.float32, device=device)

    @classmethod
    def load(cls, path: str, device: torch.device | str = torch.device("cpu")) -> "MusicSequence":
        data = np.load(path, allow_pickle=True)

        num_frames = int(data["num_frames"])
        dt = float(data["dt"])
        num_keys = data["keys"].shape[1]

        music_seq = cls(
            num_frames=num_frames,
            dt=dt,
            num_keys=num_keys,
            finger_names=data["finger_names"],
            joint_names=data["joint_names"],
            keypoint_names=data["keypoint_names"],
            device=device,
        )

        music_seq.keys[:] = torch.from_numpy(data["keys"]).to(device)
        music_seq.active_fingers[:] = torch.from_numpy(data["active_fingers"]).to(device)
        music_seq.finger_placements[:] = torch.from_numpy(data["finger_placements"]).to(device)
        music_seq.fingertip_locations[:] = torch.from_numpy(data["fingertip_locations"]).to(device)
        music_seq.joint_positions[:] = torch.from_numpy(data["joint_positions"]).to(device)
        music_seq.keypoint_locations[:] = torch.from_numpy(data["keypoint_locations"]).to(device)

        return music_seq

    def save(self, path: str) -> None:
        data = {
            "num_frames": self.num_frames,
            "dt": self.dt,
            "finger_names": self.finger_names,
            "joint_names": self.joint_names,
            "keypoint_names": self.keypoint_names,
            "keys": self.keys.cpu().numpy(),
            "active_fingers": self.active_fingers.cpu().numpy(),
            "finger_placements": self.finger_placements.cpu().numpy(),
            "fingertip_locations": self.fingertip_locations.cpu().numpy(),
            "joint_positions": self.joint_positions.cpu().numpy(),
            "keypoint_locations": self.keypoint_locations.cpu().numpy(),
        }
        np.savez(path, **data)

    @property
    def device(self) -> torch.device | str:
        return self._device

    def to(self, device: torch.device | str) -> "MusicSequence":
        self._device = device
        self.keys = self.keys.to(device)
        self.active_fingers = self.active_fingers.to(device)
        self.finger_placements = self.finger_placements.to(device)
        self.fingertip_locations = self.fingertip_locations.to(device)
        self.joint_positions = self.joint_positions.to(device)
        self.keypoint_locations = self.keypoint_locations.to(device)
        return self

    def time_to_frame(self, time: float) -> int:
        return int(time / self.dt)

    def frame_to_time(self, frame: int) -> float:
        return frame * self.dt

    def _clamp_frame_indices(self, frame_indices: torch.Tensor) -> torch.Tensor:
        return frame_indices.clamp(0, self.num_frames - 1)

    def get_frames(self, frame_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        fingerings might not be full, they need to be masked with the active keys.

        Args:
            frame_indices: The indices of the frames to get.

        Returns:
            A tuple of the active keys, active fingers, and fingerings.
            active_keys contains boolean masks of keys among the total 88 keys that should be pressed
            active_fingers contains boolean masks of fingers among the total 5 fingers that are active (pressing a key)
            fingerings contains the fingerings (which finger should press which key)
        """
        frame_indices = self._clamp_frame_indices(frame_indices)
        return (
            self.keys[frame_indices],
            self.active_fingers[frame_indices],
            self.finger_placements[frame_indices],
        )

    def get_keypoint_frames(self, frame_indices: torch.Tensor) -> torch.Tensor:
        frame_indices = self._clamp_frame_indices(frame_indices)
        return self.keypoint_locations[frame_indices]

    def scale(self, scale_factor: torch.Tensor) -> None:
        self.fingertip_locations[:] *= scale_factor
        self.keypoint_locations[:] *= scale_factor

    def translate(self, translation: torch.Tensor) -> None:
        self.fingertip_locations[:] += translation
        self.keypoint_locations[:] += translation

    def extend_frames(self, num_frames: int) -> None:
        if num_frames <= 0:
            return

        self.keys = torch.cat([self.keys, torch.zeros(num_frames, self.num_keys, dtype=torch.bool, device=self.device)], dim=0)
        self.active_fingers = torch.cat([self.active_fingers, torch.zeros(num_frames, self.num_fingers, dtype=torch.bool, device=self.device)], dim=0)
        self.finger_placements = torch.cat([self.finger_placements, torch.zeros(num_frames, self.num_fingers, dtype=torch.int32, device=self.device)], dim=0)
        self.fingertip_locations = torch.cat([self.fingertip_locations, torch.zeros(num_frames, self.num_fingers, 3, dtype=torch.float32, device=self.device)], dim=0)
        self.joint_positions = torch.cat([self.joint_positions, torch.zeros(num_frames, self.num_joints, dtype=torch.float32, device=self.device)], dim=0)
        self.keypoint_locations = torch.cat([self.keypoint_locations, torch.zeros(num_frames, self.num_keypoints, 3, dtype=torch.float32, device=self.device)], dim=0)
        self.num_frames += num_frames
        self.duration += num_frames * self.dt

    def add_note(
        self,
        key: int,
        finger: int,
        start_time: float,
        duration: float | None = None,
        end_time: float | None = None,
    ):
        start_frame = self.time_to_frame(start_time)
        if end_time is not None:
            end_frame = self.time_to_frame(end_time)
        elif duration is not None:
            end_frame = start_frame + self.time_to_frame(duration)
        else:
            raise ValueError("Either end_time or duration must be provided")

        if end_frame > self.num_frames:
            self.extend_frames(end_frame - self.num_frames)

        self.keys[start_frame:end_frame, key] = True
        self.active_fingers[start_frame:end_frame, finger] = True
        self.finger_placements[start_frame:end_frame, finger] = key

    def speedup(self, speedup: float) -> None:
        """Speed up or slow down the music sequence by the given factor.

        Values greater than 1 speed up the sequence (i.e., make it faster, shorter duration),
        values less than 1 slow down the sequence (i.e., make it slower, longer duration).
        Zero and negative values are not allowed.

        This method converts frames to times, scales the time by the speedup factor,
        creates new buffers with the adjusted size, and maps the old frame data to
        the new frames accordingly.

        Args:
            speedup: The factor by which to speed up the sequence.
        """
        if speedup <= 0:
            raise ValueError("speedup must be positive.")

        if speedup == 1.0:
            return  # No-op if factor is 1.0

        # Store old values before updating
        old_num_frames = self.num_frames

        # Calculate new duration and number of frames
        # Speedup > 1 means faster (shorter duration), speedup < 1 means slower (longer duration)
        new_duration = self.duration / speedup
        new_num_frames = int(new_duration / self.dt)

        # Create new buffers with the new size
        new_keys = torch.zeros(new_num_frames, self.num_keys, dtype=torch.bool, device=self.device)
        new_active_fingers = torch.zeros(new_num_frames, self.num_fingers, dtype=torch.bool, device=self.device)
        new_finger_placements = torch.zeros(new_num_frames, self.num_fingers, dtype=torch.int32, device=self.device)
        new_fingertip_locations = torch.zeros(new_num_frames, self.num_fingers, 3, dtype=torch.float32, device=self.device)
        new_joint_positions = torch.zeros(new_num_frames, self.num_joints, dtype=torch.float32, device=self.device)
        new_keypoint_locations = torch.zeros(new_num_frames, self.num_keypoints, 3, dtype=torch.float32, device=self.device)

        # Map old frames to new frames based on time scaling
        for new_frame in range(new_num_frames):
            # Convert new frame to time
            new_time = self.frame_to_time(new_frame)

            # Scale time back to find corresponding old time
            # If speedup > 1, we're going faster, so old_time is further ahead
            old_time = new_time * speedup

            # Convert old time to old frame index (clamp to valid range)
            old_frame = self.time_to_frame(old_time)
            old_frame = min(old_frame, old_num_frames - 1)

            # For discrete values, use nearest neighbor sampling
            new_keys[new_frame] = self.keys[old_frame]
            new_active_fingers[new_frame] = self.active_fingers[old_frame]
            new_finger_placements[new_frame] = self.finger_placements[old_frame]

            # For continuous values, use linear interpolation
            # Calculate the exact frame position (can be fractional)
            old_frame_exact = old_time / self.dt

            # Get the two frames to interpolate between
            old_frame_0_int = int(old_frame_exact)
            old_frame_0 = min(max(old_frame_0_int, 0), old_num_frames - 1)
            old_frame_1 = min(old_frame_0 + 1, old_num_frames - 1)

            # Calculate interpolation weight (alpha)
            alpha = old_frame_exact - old_frame_0_int
            alpha = max(0.0, min(alpha, 1.0))  # Clamp alpha to [0, 1]

            # If we're at the last frame or frames are the same, just copy
            if old_frame_0 == old_frame_1:
                new_fingertip_locations[new_frame] = self.fingertip_locations[old_frame_0]
                new_joint_positions[new_frame] = self.joint_positions[old_frame_0]
                new_keypoint_locations[new_frame] = self.keypoint_locations[old_frame_0]
            else:
                # Linear interpolation: (1 - alpha) * frame_0 + alpha * frame_1
                new_fingertip_locations[new_frame] = (
                    (1 - alpha) * self.fingertip_locations[old_frame_0]
                    + alpha * self.fingertip_locations[old_frame_1]
                )
                new_joint_positions[new_frame] = (
                    (1 - alpha) * self.joint_positions[old_frame_0]
                    + alpha * self.joint_positions[old_frame_1]
                )
                new_keypoint_locations[new_frame] = (
                    (1 - alpha) * self.keypoint_locations[old_frame_0]
                    + alpha * self.keypoint_locations[old_frame_1]
                )

        # Update instance variables
        self.num_frames = new_num_frames
        self.duration = new_duration
        self.keys = new_keys
        self.active_fingers = new_active_fingers
        self.finger_placements = new_finger_placements
        self.fingertip_locations = new_fingertip_locations
        self.joint_positions = new_joint_positions
        self.keypoint_locations = new_keypoint_locations
