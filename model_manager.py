import gym
import os
import threading
import copy
import time

import numpy as np
import torch
from torch import nn

from TD3_agent import TD3_Agent
from flag import build_flag
from replayer_buffer import ReplayBuffer
from utils import *


class ModelManager(nn.Module):
    def __init__(self, agent_pools, args, writer, agent_writers, writer_freq, device):
        super(ModelManager, self).__init__()
        self.models = nn.ModuleList([agent.model for agent in agent_pools])

        models_actor = nn.ModuleList([agent.model.actor for agent in agent_pools])
        models_critics = []
        for agent in agent_pools:
            models_critics.extend([agent.model.critic, agent.model.mr_critic])
        models_critics = nn.ModuleList(models_critics)

        self.args = args
        self.population_size = len(agent_pools)
        self.batch_size = args.batch_size
        self.device = device
        self.optimizer = torch.optim.Adam(
            [{'params': models_actor.parameters(),
             'lr': 1e-3},
            {'params': models_critics.parameters(), 
             'lr': 1e-3}])

        self.writer = writer
        self.agent_writers = agent_writers
        self.writer_freq = writer_freq
        self.div_metric = args.div_metric

    def update(self, train_data, iter_):
        behavior_descriptor, samples_all, guidance_sample, guide_coef = train_data
        diversity_info = self.get_diversity_info(behavior_descriptor, iter_)

        loss = 0
        for agent_id in range(self.population_size):

            loss_dict = self.models[agent_id].forward_all(samples_all[agent_id], 
                                                        diversity_info[agent_id], 
                                                        guidance_sample,
                                                        guide_coef[agent_id],
                                                        self.batch_size, 
                                                        self.device,
                                                        False,
                                                        self.writer)

            loss_tmp = loss_dict["critic_loss"] + loss_dict["mr_critic_loss"] + loss_dict["actor_loss"]
            loss += loss_tmp

            if iter_ % self.writer_freq == 0:
                self.agent_writers[agent_id].add_scalar("diversity_loss", loss_dict["diversity_loss"], iter_)
                self.agent_writers[agent_id].add_scalar("reward_loss", loss_dict["reward_loss"], iter_)
                self.agent_writers[agent_id].add_scalar("guidance_loss", loss_dict["guidance_loss"], iter_)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_diversity_info(self, behavior_descriptor, iter_):

        diversity_metric = DiversityMetric(behavior_descriptor)
        diversity_metric.run(metric=self.div_metric)

        if iter_ % self.writer_freq == 0:
            self.writer.add_scalar("diversity", diversity_metric.diversity, iter_)
            stat_tmp = torch.stack(behavior_descriptor, dim=0).detach()
            stat_tmp = torch.var(stat_tmp, dim=0)
            name = [f"action_dim_{idx}" for idx in range(stat_tmp.shape[0]-2)] + ["times", "score_rate"]
            for i in range(stat_tmp.shape[0]):
                self.writer.add_scalar(name[i], stat_tmp[i], iter_)

        return diversity_metric.behavior_descriptor