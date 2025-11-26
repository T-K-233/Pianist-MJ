from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .cfg import piano_prime_wuji_env_cfg, piano_prime_wuji_ppo_runner_cfg

register_mjlab_task(
    task_id="Piano-Prime-Wuji",
    env_cfg=piano_prime_wuji_env_cfg(),
    play_env_cfg=piano_prime_wuji_env_cfg(play=True),
    rl_cfg=piano_prime_wuji_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
