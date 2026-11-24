from omegaconf import DictConfig

from jaxolotl.alg.curriculum import wrap_for_training
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import (
    NoisySemanticLDBAWrapper,
)
from jaxolotl.alg.sem_ltl.wrappers.semantic_ldba_wrapper import SemanticLDBAWrapper
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.normalize_reward_wrapper import (
    NormalizeRewardWrapper,
)
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def wrap_env(
    env: Environment | EnvWrapper, cfg: DictConfig, training: bool
) -> EnvWrapper | Environment:
    if training:
        env = wrap_for_training(env, cfg, SemanticLDBAWrapper)
        env = NormalizeRewardWrapper(env, gamma=cfg.rl_alg.gamma)
    else:
        finite = cfg.get("eval", {}).get("finite", False)
        eval_cfg = cfg.get("eval", {})
        noise_prob = float(eval_cfg.get("noise_prob", 0.0))
        use_noisy = bool(eval_cfg.get("use_noisy_wrapper", False))
        if noise_prob > 0.0 or use_noisy:
            env = NoisySemanticLDBAWrapper(
                env, noise_prob=noise_prob, overwrite_finite=finite
            )
        else:
            env = SemanticLDBAWrapper(env, overwrite_finite=finite)
    return env
