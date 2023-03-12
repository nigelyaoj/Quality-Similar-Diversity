import torch


class DiversityMetric(object):
    def __init__(self, args, valid_num_players=None):
        self.num_players = args.num_players
        self.valid_num_players = valid_num_players if valid_num_players is not None else args.num_players
        self.b_s = []
        self.diversity = None
        self.action_dim = args.action_dim


    def set_input(self, trajs):
        for i in range(self.num_players):
            tmp = self.behavior_descriptor(trajs[i])
            self.b_s.append(tmp)

    def behavior_descriptor(self, traj):
        val = torch.zeros(self.action_dim * len(traj), dtype=torch.float32)
        for i, a in enumerate(traj):
            val[i * self.action_dim + a] = 1.0
        val.requires_grad = True
        return val

    def calculate_diversity(self):
        div_matrix = torch.zeros(
            (self.valid_num_players, self.valid_num_players))
        for i in range(self.valid_num_players):
            for j in range(self.valid_num_players):
                div_matrix[i][j] = self.kernel_func(self.b_s[i], self.b_s[j])
        return torch.det(div_matrix)
        # return  torch.sum(div_matrix)

    def kernel_func(self, a, b):
        return torch.exp(-((a - b) ** 2 / 10).sum())
        # return ((a - b) ** 2).sum()

    def eval(self, trajs):
        self.clean()
        self.set_input(trajs)
        self.diversity = self.calculate_diversity()
        self.diversity.backward()

        self.pair_dist = 0

    def clean(self):
        self.b_s = []
        self.diversity = None
