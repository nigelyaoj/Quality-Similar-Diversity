import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, lstm_hidden_dim, results_dim, traj_len, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.lstm_hidden_0 = np.zeros((max_size, lstm_hidden_dim))
        self.lstm_hidden_1 = np.zeros((max_size, lstm_hidden_dim))
        self.state_act_traj = np.zeros((max_size, traj_len, state_dim + action_dim))
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.results = np.zeros((max_size, results_dim))
        self.not_done = np.zeros((max_size, 1))

        self.output_device = device
        self.dtype = torch.float32


    def add(self, state, action, next_state, reward, results, done, lstm_hidden_cell, state_act_traj):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.results[self.ptr] = results
        self.not_done[self.ptr] = 1. - done

        self.lstm_hidden_0[self.ptr] = lstm_hidden_cell[0][0].cpu().numpy()
        self.lstm_hidden_1[self.ptr] = lstm_hidden_cell[1][0].cpu().numpy()

        self.state_act_traj[self.ptr] = state_act_traj

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.from_numpy(self.state[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.action[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.next_state[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.reward[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.results[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.not_done[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.lstm_hidden_0[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.lstm_hidden_1[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.state_act_traj[ind]).to(self.dtype).to(self.output_device),
        )
    
    def sample_terminate(self, batch_size):
        indexs = np.where(self.not_done[:self.size] == 0)[0]
        ind = np.random.choice(indexs, size=batch_size)
        return (
            torch.from_numpy(self.state[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.action[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.next_state[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.reward[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.results[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.not_done[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.lstm_hidden_0[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.lstm_hidden_1[ind]).to(self.dtype).to(self.output_device),
            torch.from_numpy(self.state_act_traj[ind]).to(self.dtype).to(self.output_device),
        )