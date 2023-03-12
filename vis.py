import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import matplotlib
matplotlib.use('TkAgg')
np.random.seed(321)


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
        self.div_matrix = div_matrix
        if metric == "DPP":
            tmp = torch.det(div_matrix)
            return tmp
            if torch.abs(tmp) > 1000:
                print("Warning")
            return torch.clamp(torch.exp(tmp), -1000, 1000)
        elif metric == "MSE":
            return torch.mean(div_matrix)
        else:
            raise NotImplementedError

    def kernel_func(self, a, b, metric):
        if metric == "DPP":
            return torch.exp(-torch.abs(a - b).sum()/2)
        elif metric == "MSE":
            return torch.sqrt(((a - b) ** 2).mean())
        else:
            raise NotImplementedError


def eval_diversity(behavior_descriptor):

    diversity_metric = DiversityMetric(torch.tensor(behavior_descriptor))
    diversity_metric.run(backward=False)
    return diversity_metric.diversity.data


def getpoints_ways2(eval_res, start, end, ENV, population_size=8):
    skip = (end - start) // 10
    eval_res_all = eval_res

    eval_res_reward = np.array(list(eval_res_all.keys()))

    data_points = []
    for st in np.arange(start, end+1, skip):
        indexs = np.where((eval_res_reward > st) & (
            eval_res_reward < st + skip))[0]
#         print(len(indexs))
        if len(indexs) == 0:
            data_points.append([st, 0])
            continue
        diversity = []
        num_samples = 100
        for ii in range(num_samples):
            samples = np.random.choice(indexs, replace=(
                len(indexs) <= population_size), size=population_size)
            behavior_descriptor = [
                eval_res_all[eval_res_reward[idx]] for idx in samples]
            div = eval_diversity(behavior_descriptor)
            diversity.append(div.data)
        data_points.append([st, np.max(diversity)])

    return np.array(data_points)[:, 0], np.array(data_points)[:, 1]


end_lookup = {"Humanoid": 5500, "Hopper": 3500, "Walker2d": 5000}
dataframes = []

exp_num = {}
for ENV in ["Humanoid", "Hopper", "Walker2d"]:
    end = end_lookup[ENV]
    method = "QSD"

    exp_data = []
    for file_ in os.listdir(f"{ENV}"):
        if file_.find(method) > -1:
            tmp = exp_num.setdefault(ENV, {}).get(method, 0)
            exp_num[ENV][method] = tmp+1
            eval_res = pickle.load(open(f"{ENV}/{file_}", "rb"))
            eval_res_new = {}
            for val in eval_res:
                eval_res_new.update(val)
            eval_res = eval_res_new

            # the range of action is (-0.4, 0.4) not (-1, 1) as in [Hopper, Walker2d], renormalize it
            if ENV == 'Humanoid':
                for val in eval_res.values():
                    val[:17] = val[:17] / 0.16
            exp_data.append(eval_res)

    for ii in range(len(exp_data)):
        eval_res = exp_data[ii]
        data_x, data_y = getpoints_ways2(eval_res,  start=0, ENV=ENV, end=end)
        df = pd.DataFrame({
            "quality": data_x,
            "diversity": data_y,
            "exp_num": np.ones(len(data_x)) * ii,
            "env": [ENV] * len(data_x),
            "method": [method] * len(data_x)
        })
        dataframes.append(df)


dataframes = pd.concat(dataframes, axis=0)

_color = 'tab:red'
_marker = ':x'
plt.rcParams['font.sans-serif'] = 'arial'

pad_left = 0.05
pad_right = 0.005
pad = 0.04
num_width = 3
width = (1 - pad * (num_width - 1) - pad_left - pad_right) / num_width
fig = plt.figure(figsize=(12, 3.3))

yticks = [
    [0, 0.15,  0.3, 0.45],
    [0, 0.08, 0.16, 0.24],
    [0, 0.08, 0.16, 0.24],
]

xticks = [
    [1500, 3000, 4500, 6000],
    [1000, 2000, 3000, ],
    [1000, 2000, 3000, 4000, 5000],
]


def fill_zero(df):
    m = df['diversity'].mean()
    df['diversity'][df["diversity"] == 0] = m
    return df


for plot_i, ENV in enumerate(["Humanoid", "Hopper", "Walker2d"]):

    df = dataframes[dataframes['env'] == ENV]
    df = df.groupby(["method", "quality", ])["diversity"].agg(["mean", "std"])
    df = df.reset_index()

    ax = plt.axes([plot_i * (width + pad) + pad_left, 0.14, width, 0.78])

    quality = df["quality"]
    div_mean = df["mean"]
    div_std = df["std"] / np.sqrt(len(exp_data))
    re_label = "QSD-PBT (Ours)"

    ax.errorbar(quality[1:], div_mean[1:], yerr=div_std[1:],
                fmt=_marker,
                capsize=2,
                elinewidth=1,
                capthick=1,
                color=_color, markersize=5, markerfacecolor="w", clip_on=False, label=re_label)

    ax.legend(fontsize=8)
    ax.set_xlabel('Quality')
    if plot_i == 0:
        ax.set_ylabel('Diversity')

    ax.grid(linestyle=':', axis='y')
    ax.set_xticks(xticks[plot_i])
    ax.set_yticks(yticks[plot_i])
    ax.set_title(ENV)
plt.savefig("fig_mujoco.png", dpi=300)


print("===score > 0%===")
df = dataframes.groupby(['env', 'method', 'exp_num'])[
    'diversity'].agg("sum").reset_index()
df = df.groupby(['env', 'method'])['diversity'].agg(
    ['mean', 'std']).reset_index()

for ii in range(df.shape[0]):
    print(df['env'].values[ii],
          df['method'].values[ii],
          np.round(df['mean'].values[ii], 4),
          np.round(df['std'].values[ii], 4)
          )


print("===score > 60%===")


def fit(df):
    env = df['env'].values[0]
    r_max = end_lookup[env]
    df = df[df['quality'] > 0.6*r_max]
    return df


df = dataframes.groupby(['env'], as_index=False).apply(fit)
df = df.groupby(['env', 'method', 'exp_num'])[
    'diversity'].agg("sum").reset_index()
df = df.groupby(['env', 'method'])['diversity'].agg(
    ['mean', 'std']).reset_index()

for ii in range(df.shape[0]):
    print(df['env'].values[ii],
          df['method'].values[ii],
          np.round(df['mean'].values[ii], 4),
          np.round(df['std'].values[ii], 4)
          )
