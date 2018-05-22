#!/usr/bin/env python3

from baselines import logger
from baselines.common import tf_util as U
from baselines.ppo1 import cal_policy, pposgd_cal
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.ppo1.knowledge import knowledge_dict
from baselines.ppo1.knowledge import network_config


def train(config, env_id, num_timesteps, seed):
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, config, ob_space, ac_space):
        return cal_policy.CalPolicy(name=name, config=config, ob_space=ob_space, ac_space=ac_space)

    env = make_mujoco_env(env_id, seed)
    pposgd_cal.learn(config, env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()


def gen_config():
    # only using config
    config = dict()
    config['run_type'] = 'train'
    config['continue'] = False
    # construction configuration:
    config['env_type'] = 'reacher'
    config['update_name'] = 'reacher'
    # network config:
    network_config(config)
    return config


def main():
    logger.configure()
    config = gen_config()
    train(config, config['environment'], num_timesteps=1e6, seed=0)


if __name__ == '__main__':
    main()
