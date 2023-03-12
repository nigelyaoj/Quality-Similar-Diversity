import numpy as np
import torch
import time
import copy
import os
import logging
from logging import handlers
from tensorboardX import SummaryWriter
import gym
from actor_critic import *
from replayer_buffer import ReplayBuffer



class Agent_Model(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, lstm_hidden_dim, results_dim, device, args):
        super(Agent_Model, self).__init__()

        # Actor
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        for param in self.actor_target.parameters():
            param.requires_grad = False
        # Reward Critic
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        for param in self.critic_target.parameters():
            param.requires_grad = False
        # Multi Results Critic
        self.mr_critic = MultiResults_Critic(
            state_dim, action_dim, lstm_hidden_dim, results_dim).to(device)
        self.mr_critic_target = copy.deepcopy(self.mr_critic)
        for param in self.mr_critic_target.parameters():
            param.requires_grad = False

        self.max_action = max_action
        self.discount = args.discount
        self.res_discount = args.res_discount
        self.tau = args.tau
        self.loss_weight_guide = args.loss_weight_guide
        self.loss_weight_lambda = args.loss_weight_lambda
        # Target policy smoothing is scaled wrt the action scale
        self.policy_noise = args.policy_noise * max_action
        self.noise_clip = args.noise_clip * max_action
        self.policy_freq = args.policy_freq

        self.total_it = -1

    def forward_action(self, state):
        return self.actor(state)

    def forward_all(self, samples, div_info, guidance_sample, guide_coef, batch_size, device, is_main_agent, writer):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, results, not_done, lstm_hidden_0, lstm_hidden_1, state_act_traj = samples

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value for reward
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

            # Compute the multi head res
            state_act_info = torch.cat(
                [next_state, next_action], dim=1).unsqueeze(1)
            state_act_traj_next = torch.cat(
                [state_act_traj, state_act_info], dim=1)
            multi_head_res = results + not_done * self.res_discount * self.mr_critic_target(
                next_state, next_action, (lstm_hidden_0, lstm_hidden_1), state_act_traj_next)

        loss_dict = {}
        # update Q values estimates
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)
        loss_dict["critic_loss"] = critic_loss

        # update multi results estimates
        current_res = self.mr_critic(
            state, action, (lstm_hidden_0, lstm_hidden_1), state_act_traj)
        mr_critic_loss = F.mse_loss(current_res, multi_head_res)
        loss_dict["mr_critic_loss"] = mr_critic_loss

        if self.total_it % self.policy_freq == 0:
            action = self.actor(state)
            # get diversity reward
            diversity_loss = - (div_info.grad.detach() * self.mr_critic_target(
                state, action, (lstm_hidden_0, lstm_hidden_1), state_act_traj)).sum(1).mean()
            # for mujoco: $\lambda = \lambda_0 * exp (-t/t_0)$
            diversity_loss = diversity_loss * \
                self.loss_weight_lambda * np.exp(- self.total_it / 2e6)
            loss_dict["diversity_loss"] = diversity_loss.data
            # get quality reward
            reward_loss = -self.critic_target.Q1(state, action).mean()
            loss_dict["reward_loss"] = reward_loss.data
            # get guidance reward
            # add guidance: inspired by https://arxiv.org/abs/2001.02907
            if guidance_sample is not None:
                state_guide, action_guide = guidance_sample[0], guidance_sample[1]
                guidance_loss = F.mse_loss(action_guide, self.actor(
                    state_guide)) * self.loss_weight_guide * guide_coef
                loss_dict["guidance_loss"] = guidance_loss.data
            else:
                guidance_loss = 0
                loss_dict["guidance_loss"] = guidance_loss

            actor_loss = diversity_loss + reward_loss + guidance_loss

            loss_dict["actor_loss"] = actor_loss
        else:
            loss_dict["actor_loss"] = 0

        if self.total_it % self.policy_freq == 0:
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.mr_critic.parameters(), self.mr_critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss_dict


class TD3_Agent():
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lstm_hidden_dim,
        args,
        agent_id,
        device
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.results_dim = action_dim + 2

        self.model = Agent_Model(
            state_dim, action_dim, max_action, lstm_hidden_dim, self.results_dim, device, args)
        self.device = args.device
        self.dtype = torch.float32
        self.env = gym.make(args.env)

        self.total_it = 0
        self.sample_it = 0
        self.running_reward = 0
        self.running_reward_moment = 0.9

        self.agent_id = agent_id
        self.prefix = os.path.join(args.log_path, args.exp_name,
                                   f"{args.policy}_{args.env}_p{args.population_size}_lambda{args.loss_weight_lambda}_guide{args.loss_weight_guide}_seed{args.seed}")

        # self.replay_buffer = ReplayBuffer(state_dim, action_dim, device)
        self.start_timesteps = args.start_timesteps
        self.max_timesteps = args.max_timesteps
        self.episode_num = 0

        self.expl_noise = args.expl_noise

        # summary
        writer_path = os.path.join(
            self.prefix, f"tb_summary/actor/{self.agent_id}")
        if not os.path.exists(writer_path):
            os.makedirs(writer_path)
        self.writer = SummaryWriter(writer_path)

        # logger
        if not os.path.exists(os.path.join(self.prefix, 'train_log')):
            os.makedirs(os.path.join(self.prefix, 'train_log'))
        output_file = os.path.join(
            self.prefix, f'train_log/train_{self.agent_id}.log')
        self.logger = self.get_logger(output_file)

        self.learning_delay = 1

    def get_logger(self, output_file):
        logger = logging.getLogger(f'train-{self.agent_id}')
        logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(output_file, encoding='utf-8')
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

        return logger

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.model.forward_action(state).cpu().data.numpy().flatten()

    def load(self, filename=None):

        self.critic.load_state_dict(torch.load(
            filename + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(
            filename + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.mr_critic.load_state_dict(torch.load(
            filename + "_multi_results_critic", map_location=self.device))
        self.mr_critic_optimizer.load_state_dict(torch.load(
            filename + "_multi_results_critic_optimizer", map_location=self.device))
        self.mr_critic_target = copy.deepcopy(self.mr_critic)

        self.actor.load_state_dict(torch.load(
            filename + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(
            filename + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)

    def check_model_updates(self,):
        pass

    def play_game(self, iter_,
                  start_timesteps,
                  max_timesteps,
                  replay_buffer,
                  traj_len):

        env = self.env
        state, done = env.reset(), False
        state_act_trajectory = [
            np.zeros(self.state_dim + self.action_dim) for _ in range(traj_len)]
        lstm_hidden_cell = (torch.zeros((1, 1, self.model.mr_critic_target.lstm_hidden_dim)).to(self.dtype).to(self.device),
                            torch.zeros((1, 1, self.model.mr_critic_target.lstm_hidden_dim)).to(self.dtype).to(self.device))
        episode_reward = 0
        episode_timesteps = 0
        bd_stats = BehaviorStat(self.action_dim, self.max_action)

        for t in range(int(max_timesteps)):
            while self.total_it * 10 < self.sample_it:
                print("waiting", self.total_it, self.sample_it)
                time.sleep(0.1)
            episode_timesteps += 1
            # Select action randomly or according to policy
            if t < start_timesteps:
                action = env.action_space.sample()
            else:
                self.sample_it += 1
                with torch.no_grad():
                    action = (
                        self.select_action(np.array(state))
                        + np.random.normal(0, self.max_action *
                                           self.expl_noise, size=self.action_dim)
                    ).clip(-self.max_action, self.max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action)

            state_act_info = np.concatenate([state, action])
            if episode_timesteps >= 5:
                with torch.no_grad():
                    tmp = torch.from_numpy(
                        state_act_trajectory[-traj_len]).to(self.dtype).to(self.device)
                    _, lstm_hidden_cell = self.model.mr_critic_target.lstm(
                        tmp.unsqueeze(0).unsqueeze(1), lstm_hidden_cell)
            state_act_trajectory.append(state_act_info)
            bd_stats.update_stats(state, action, reward)
            done_bool = float(done)

            results = np.zeros(self.results_dim)

            if done:
                results = bd_stats.get_behavior_discriptor()
                bd_stats = BehaviorStat(self.action_dim, self.max_action)

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state,
                              reward, results, done_bool,
                              lstm_hidden_cell,
                              state_act_trajectory[-traj_len:])

            state = next_state
            episode_reward += reward

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                self.logger.info(
                    f"Agent: {self.agent_id} Total T: {iter_*max_timesteps+t+1} Episode Num: {self.episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                self.running_reward = self.running_reward_moment * \
                    self.running_reward + \
                    (1-self.running_reward_moment) * episode_reward
                # Reset environment
                state, done = env.reset(), False
                state_act_trajectory = [
                    np.zeros(self.state_dim + self.action_dim) for _ in range(traj_len)]
                lstm_hidden_cell = (torch.zeros((1, 1, self.model.mr_critic_target.lstm_hidden_dim)).to(self.dtype).to(self.device),
                                    torch.zeros((1, 1, self.model.mr_critic_target.lstm_hidden_dim)).to(self.dtype).to(self.device))
                episode_reward = 0
                episode_timesteps = 0
                self.episode_num += 1

    def eval_play(self, episode_num=10, print_info=True):
        eval_rewards = []
        eval_results = []

        env = self.env
        for epi_num in range(episode_num):
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            bd_stats = BehaviorStat(self.action_dim, self.max_action)
            while not done:
                with torch.no_grad():
                    action = (
                        self.select_action(np.array(state))
                        + np.random.normal(0, self.max_action *
                                           self.expl_noise, size=self.action_dim)
                    ).clip(-self.max_action, self.max_action)

                # perform action
                next_state, reward, done, _ = env.step(action)
                bd_stats.update_stats(state, action, reward)
                state = next_state

                episode_timesteps += 1
                episode_reward += reward

                if done:
                    results = bd_stats.get_behavior_discriptor()
                    eval_rewards.append(episode_reward)
                    eval_results.append(results)
                    if print_info:
                        print(
                            f"Agent: {self.agent_id} Episode Num: {epi_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

        eval_info = {"reward": np.mean(eval_rewards),
                     "results": np.mean(np.array(eval_results), axis=0),
                     "eval_nums": episode_num}

        return eval_info


class BehaviorStat(object):
    def __init__(self, action_dim, max_action):
        self.stats = {"times": 0,
                      "score": 0,
                      "action": np.zeros(action_dim)}
        self.max_action = max_action

    def update_stats(self, state, action, reward):
        self.stats["times"] += 1
        self.stats["score"] += reward
        self.stats["action"] += np.array(action)

    def get_behavior_discriptor(self):

        act_bd = self.stats["action"] / self.stats["times"]
        # normalize
        act_bd = (act_bd + self.max_action) / 2 * (self.max_action)
        # TODO '1000' '5' hard code for timestep and score
        times = self.stats["times"] / 1000
        score_rate = self.stats["score"] / self.stats["times"] / 5

        return np.concatenate([act_bd, np.array([times, score_rate])])
