from utils import *
from model_manager import ModelManager
from replayer_buffer import ReplayBuffer
from flag import build_flag
from TD3_agent import TD3_Agent
import matplotlib.pyplot as plt
import threading
import copy
import os
import time
import pickle

import numpy as np
import torch
from torch import nn
import gym
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')


class Population_Trainer(object):
    def __init__(self, env, args):

        # base config
        self.args = args
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.traj_len = args.traj_len
        self.env = env
        self.population_size = args.population_size
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        print("=" * 10, self.state_dim, self.action_dim)
        self.results_dim = self.action_dim + 2
        self.device = args.device
        self.init_agents()

        # log
        self.prefix = self.agent_pools[0].prefix
        writer_path = os.path.join(self.prefix, f"tb_summary/learner")
        self.writer = SummaryWriter(writer_path)
        self.writer_freq = 100

        # train
        self.total_iter = 100000000
        self.max_timesteps = self.env._max_episode_steps * \
            5  # max timesteps each iteration
        self.warmup_timesteps = [self.max_timesteps] * \
            self.population_size  # warm up timesteps
        self.batch_size = args.batch_size
        self.save_freq = args.save_freq
        agent_writers = [agent.writer for agent in self.agent_pools]
        self.model_manager = ModelManager(
            self.agent_pools, self.args, self.writer, agent_writers, self.writer_freq, self.device)
        self.reward_threshold_to_guide = args.reward_threshold_to_guide

    def init_agents(self):
        kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": float(self.env.action_space.high[0]),
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "args": self.args,
            "device": self.device
        }

        self.agent_pools = []
        if self.args.policy == "TD3":
            for agent_id in range(self.population_size):
                kwargs["agent_id"] = agent_id
                agent = TD3_Agent(**kwargs)
                self.agent_pools.append(agent)
        else:
            assert False, "No implement for other policy"

        self.replay_buffer_pools = []
        for agent_id in range(self.population_size):
            rb = ReplayBuffer(self.state_dim, self.action_dim, self.lstm_hidden_dim,
                              self.results_dim, self.traj_len, self.device)
            self.replay_buffer_pools.append(rb)

    def load_population_models(self, iter_):
        save_path = os.path.join(self.prefix, "model")
        filename = os.path.join(save_path, f"checkpoint_{iter_}.pt")
        self.model_manager.models.load_state_dict(
            torch.load(filename, map_location=self.device))

    def single_agent_roll_out(self, agent_id):
        eval_info = self.agent_pools[agent_id].eval_play(5, print_info=False)
        key_ = eval_info["reward"]
        value_ = eval_info["results"]
        self.eval_res[agent_id][key_] = value_

    def eval(self, start_cpt_id, end_cpt_id, skip):
        eval_res_file = os.path.join(
            self.prefix, f"eval_res_{start_cpt_id}_{end_cpt_id}_{skip}.pkl")
        if not os.path.exists(eval_res_file):
            self.eval_res = [{} for _ in range(self.population_size)]
            for iter_ in tqdm(np.arange(start_cpt_id, end_cpt_id, skip)):
                try:
                    self.load_population_models(iter_)
                except:
                    print("can not find the model")
                    break
                thread_list = []
                for agent_id in range(self.population_size):
                    thread_ = threading.Thread(
                        target=self.single_agent_roll_out, args=(agent_id,))
                    thread_list.append(thread_)
                for thread_ in thread_list:
                    thread_.start()
                for thread_ in thread_list:
                    thread_.join()

            eval_res = self.eval_res
            with open(eval_res_file, "wb") as f:
                pickle.dump(eval_res, f)
        else:
            print("eval results exist. ")


if __name__ == "__main__":

    args = build_flag()
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    env = gym.make(args.env)
    set_seed(env, args.seed)

    population_trainer = Population_Trainer(env, args)
    population_trainer.eval(start_cpt_id=1000, end_cpt_id=200000+1, skip=500)
