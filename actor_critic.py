import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class MultiResults_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, lstm_hidden_dim, results_dim):
        super(MultiResults_Critic, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        # Q1 architecture
        self.l1 = nn.Linear(self.lstm_hidden_dim, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, results_dim)

        self.lstm = nn.LSTM(state_dim + action_dim,
                            self.lstm_hidden_dim, batch_first=True)
        self.lstm.flatten_parameters()

    def forward(self, state, action, hidden_cell, state_act_traj):
        hidden_cell = (hidden_cell[0].unsqueeze(0),
                       hidden_cell[1].unsqueeze(0))
        output, _ = self.lstm(state_act_traj, hidden_cell)
        x = output[:, -1, :]
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x
