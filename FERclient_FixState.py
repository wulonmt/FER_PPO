import gymnasium as gym
import torch as th
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

import argparse
from utils.Ptime import Ptime

import flwr as fl
from collections import OrderedDict
import os
import sys
import Env
from utils.CustomPPO import CustomPPO
from utils.init_pos_config import get_init_pos, assert_alarm

def paser_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_name", help="modified log name", type=str, default ="auto")
    parser.add_argument("-s", "--save_log", help="whether save log or not", type=str, default = "True") #parser can't pass bool
    parser.add_argument("-e", "--environment", help="which my- env been used", type=str, required=True)
    # parser.add_argument("-e", "--environment", help="which my- env been used", type=str, default="Pendulum-v0")
    parser.add_argument("-t", "--train", help="training or not", type=str, default = "True")
    parser.add_argument("-r", "--render_mode", help="h for human & r for rgb_array", type=str, default = "r")
    parser.add_argument("-i", "--index", help="client index", type=int, default = 0, required = True)
    parser.add_argument("-p", "--port", help="local port", type=str, default="8080")
    parser.add_argument("-m", "--time_step", help="training time steps", type=int, default=5e3)
    return parser.parse_args()

class FixPosClient(fl.client.NumPyClient):
    def __init__(self, client_index, args=None):
        
        rm = "rgb_array" if args.render_mode == "r" else "human"

        n_cpu = 2
        batch_size = 64
        #self.env = gym.make(f"my-{args.environment}-v0", render_mode=rm)
        # self.env = make_vec_env(f"{args.environment}", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        if(args.index < 4):
            self.env = make_vec_env(f"{args.environment}", n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs = get_init_pos(args.environment, args.index))
            # self.env = make_vec_env(f"{args.environment}", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        else:
            self.env = make_vec_env(f"{args.environment}", n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs = get_init_pos(args.environment, args.index))
            # self.env = make_vec_env(f"{args.environment}", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        self.tensorboard_log=f"{args.environment}_ppo/" if args.save_log == "True" else None
        time_str = Ptime()
        time_str.set_time_now()
        if args.save_log == "True":
            self.tensorboard_log = "multiagent/" + time_str.get_time_to_hour() + "/" + self.tensorboard_log
        trained_env = self.env
        self.model = CustomPPO("MlpPolicy",
                    trained_env,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])),
                    n_steps=batch_size * 4 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=10,
                    regul_update_interval = 10,
                    learning_rate=5e-4,
                    gamma=0.99,
                    verbose=1,
                    target_kl=0.2,
                    ent_coef=0.,
                    kl_coef=0.1,
                    vf_coef=0.8,
                    tensorboard_log=self.tensorboard_log,
                    use_advantage = True,
                    tau = 0.8,
                    kl_coef_decay = 1e-6,
                    device = "cuda:0",
                    )

        self.n_round = int(0)
        self.args = args
        
        if args.save_log == "True":
            description = args.log_name if args.log_name != "auto" else \
                        f"multiagent_targetkl{self.model.target_kl:.1e}_entcoef{self.model.ent_coef:.1e}_klcoef{self.model.kl_coef:.1e}"
            self.log_name = f"{client_index}_{description}"
        else:
            self.log_name = None
        
        
    def get_parameters(self, config):
        # print(self.model.policy)
        # print(self.model.policy.state_dict().keys())
        # print([key for key, value in self.model.policy.state_dict().items() if "policy_net" in key])
        # breakpoint()
        policy_state = [value.cpu().numpy() for key, value in self.model.policy.state_dict().items()]
        # policy_state = [value.cpu().numpy() for key, value in self.model.policy.state_dict().items() if "policy_net" in key]
        return policy_state

    def set_parameters(self, parameters):
        # -----specific key-----
        # parameters = [th.tensor(v) for v in parameters]
        # features_extractor_keys = [key for key in self.model.policy.state_dict().keys() if "policy_net" in key]
        # params_dict = zip(features_extractor_keys, parameters)
        # state_dict = self.model.policy.state_dict()
        # state_dict.update(params_dict)
        # -----all parameters-----
        params_dict = zip(self.model.policy.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: th.tensor(v) for k, v in params_dict})
        # -----set to policy-----
        self.model.policy.load_state_dict(state_dict, strict=True)
        self.model.regul_policy.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.n_round += 1
        self.set_parameters(parameters)
        if("learning_rate" in config.keys()):
            self.model.learning_rate = config["learning_rate"]
        print(f"Training learning rate: {self.model.learning_rate}")
        # Train the agent
        self.model.learn(total_timesteps=self.args.time_step,
                         tb_log_name=(self.log_name + f"/round_{self.n_round}" if self.n_round>9 else self.log_name + f"/round_0{self.n_round}") if self.log_name is not None else None ,
                         reset_num_timesteps=False,
                         )
        # Save the agent
        if self.args.save_log == "True":
            print("log name: ", self.tensorboard_log + self.log_name)
            self.model.save(self.tensorboard_log + self.log_name + "/model")
            
        return self.get_parameters(config={}), self.model.num_timesteps, {}

    def evaluate(self, parameters, config):
        print("evaluating model")
        self.set_parameters(parameters)
        print("after set parameters")
        reward_mean, reward_std = evaluate_policy(self.model, self.env)
        print("after evaluate policy")
        return -reward_mean, self.model.num_timesteps, {"reward mean": reward_mean, "reward std": reward_std}

def main():
    args = paser_argument()
    assert_alarm(args.environment)

    # Start Flower client
    #port = 8080 + args.index
    client = FixPosClient(args.index, args=args)
    fl.client.start_client(
        server_address=f"127.0.0.1:" + args.port,
        client=client.to_client(),
    )
    # sys.exit()

    if args.index < 4:
        env = gym.make(args.environment, render_mode="human", **get_init_pos(args.environment, args.index))
    else:
        env = gym.make(args.environment, render_mode="human", **get_init_pos(args.environment, 4))
    # env = gym.make(args.environment, render_mode="human")

    while True:
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = client.model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

if __name__ == "__main__":
    main()
    # test()

