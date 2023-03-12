from tqdm import tqdm
import matplotlib.pyplot as plt
import threading
import copy
import os
import time

import numpy as np
import torch
from torch import nn
import gym
from tensorboardX import SummaryWriter

from TD3_agent import TD3_Agent
from flag import build_flag
from replayer_buffer import ReplayBuffer
from model_manager import ModelManager
from utils import *
import matplotlib
matplotlib.use('Agg')


class Population_Trainer(object):
    def __init__(self, env, args):

        # base config
        self.args = args
        self.traj_len = args.traj_len
        self.env = env
        self.population_size = args.population_size
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        print("=" * 10, self.state_dim, self.action_dim)
        self.results_dim = self.action_dim + 2
        self.device = args.device

        self.lstm_hidden_dim = args.lstm_hidden_dim 
        self.init_agents()

        # log
        self.prefix = self.agent_pools[0].prefix
        writer_path = os.path.join(self.prefix, f"tb_summary/learner")
        self.writer = SummaryWriter(writer_path)
        self.writer_freq = 100

        # train
        self.total_iter = 200000
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
            raise NotImplementedError

        self.replay_buffer_pools = []
        for agent_id in range(self.population_size):
            rb = ReplayBuffer(self.state_dim, self.action_dim, self.lstm_hidden_dim,
                              self.results_dim, self.traj_len, self.device)
            self.replay_buffer_pools.append(rb)

    def run(self):
        actor = threading.Thread(target=self.agents_play)
        learner = threading.Thread(target=self.train_agents)
        actor.start()
        learner.start()
        actor.join()
        learner.join()

    def agents_play(self,):
        thread_list = []
        for agent_id in range(self.population_size):
            thread_ = threading.Thread(
                target=self.single_agent_play, args=(agent_id,))
            thread_list.append(thread_)
        for thread_ in thread_list:
            thread_.start()
        for thread_ in thread_list:
            thread_.join()

    def single_agent_play(self, agent_id):
        for iter_ in range(self.total_iter):
            self.agent_pools[agent_id].play_game(iter_,
                                                 self.warmup_timesteps[agent_id],
                                                 self.max_timesteps,
                                                 self.replay_buffer_pools[agent_id],
                                                 self.traj_len)
            self.warmup_timesteps[agent_id] = 0

    def train_agents(self, ):
        while sum(self.warmup_timesteps) > 0:
            time.sleep(0.1)
            continue

        for iter_ in (range(self.total_iter)):
            train_data = self.get_train_data()
            self.model_manager.update(train_data, iter_)
            for agent in self.agent_pools:
                agent.total_it += 1

            if iter_ % self.writer_freq == 0:
                for agent in self.agent_pools:
                    agent.writer.add_scalar(
                        "Reward", agent.running_reward, iter_)

            if iter_ % self.save_freq == 0:
                self.save_population_models(iter_)

    def get_train_data(self,):
        # behavior_descriptor
        behavior_descriptor = []
        for agent_id in range(self.population_size):
            samples = self.replay_buffer_pools[agent_id].sample_terminate(
                self.batch_size)
            results, not_done = samples[-5], samples[-4]

            results = results[(not_done == 0).squeeze()].mean(0)
            behavior_descriptor.append(results)

        samples_all = []
        for agent_id in range(self.population_size):
            samples = self.replay_buffer_pools[agent_id].sample(
                self.batch_size)
            samples_all.append(samples)

        agent_rewards = [agent.running_reward for agent in self.agent_pools]
        agent_rewards = np.array(agent_rewards)
        best_agent_index = np.argmax(agent_rewards)
        best_reward = agent_rewards[best_agent_index]

        guide_coef = np.exp(- agent_rewards / best_reward)
        is_guided = agent_rewards < (
            best_reward * self.reward_threshold_to_guide)
        guide_coef *= is_guided

        guidance_sample = self.replay_buffer_pools[best_agent_index].sample(
            self.batch_size)

        return behavior_descriptor, samples_all, guidance_sample, guide_coef

    def save_population_models(self, iter_):
        save_path = os.path.join(self.prefix, "model")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, f"checkpoint_{iter_}.pt")
        torch.save(self.model_manager.models.state_dict(), filename)

    def load_population_models(self, iter_):
        save_path = os.path.join(self.prefix, "model")
        filename = os.path.join(save_path, f"checkpoint_{iter_}.pt")
        self.model_manager.models.load_state_dict(
            torch.load(filename, map_location=self.device))


if __name__ == "__main__":

    args = build_flag()
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    env = gym.make(args.env)
    set_seed(env, args.seed)

    population_trainer = Population_Trainer(env, args)
    population_trainer.run()
