import torch
import numpy as np

# Set seeds
def set_seed(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


class DiversityMetric(object):
    def __init__(self, behavior_descriptor):
        self.num_agents = len(behavior_descriptor)
        self.behavior_descriptor = behavior_descriptor
        self.device = behavior_descriptor[0].device

        for val in self.behavior_descriptor:
            val.requires_grad = True

    def run(self, metric="MSE", backward=True):
        self.diversity = self.calculate_diversity(metric)
        if backward:
            self.diversity.backward()

    def calculate_diversity(self, metric):

        div_matrix = torch.zeros(
            (self.num_agents, self.num_agents), device=self.device)
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                div_matrix[i][j] = self.kernel_func(
                    self.behavior_descriptor[i], self.behavior_descriptor[j], metric)

        if metric == "DPP":
            return torch.det(div_matrix)
        elif metric == "MSE":
            return torch.sum(div_matrix)
        else:
            raise NotImplementedError

    def kernel_func(self, a, b, metric):
        if metric == "DPP":
            return torch.exp(-((a - b) ** 2 / 10).sum())
        elif metric == "MSE":
            return ((a - b) ** 2).sum()
        else:
            raise NotImplementedError
