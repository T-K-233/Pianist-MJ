from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .cfg import prime_reach_env_cfg, prime_reach_ppo_runner_cfg

register_mjlab_task(
    task_id="Reach-Prime",
    env_cfg=prime_reach_env_cfg(),
    play_env_cfg=prime_reach_env_cfg(play=True),
    rl_cfg=prime_reach_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
