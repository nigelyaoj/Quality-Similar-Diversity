import os
import numpy as np
import torch
import gym
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import animation
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import cv2
from utils import *
from flag import build_flag
from TD3_agent import TD3_Agent, BehaviorStat
from replayer_buffer import ReplayBuffer

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def save_to_video(frames, path='./', filename='gym_animation.avi'):
    fps = 30 
    size = (frames[0].shape[1], frames[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter(filename,fourcc,fps,size)
    for frame in frames:
        videoWriter.write(frame)
    videoWriter.release()

def eval_policy(policy, eval_env, seed, eval_episodes=10, save_video_name="video.avi"):
    
    bd_stat = BehaviorStat(eval_env.action_space.shape[0], float(eval_env.action_space.high[0]))
    
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        epi_reward = 0.
        state, done = eval_env.reset(), False

        frames = []
        while not done:
            action = policy.select_action(np.array(state))
            # import pdb;pdb.set_trace()
            next_state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            epi_reward += reward
            # eval_env.render() 
            # time.sleep(0.01)
            frames.append(env.render(mode="rgb_array"))
            import pdb;pdb.set_trace()
            bd_stat.update_stats(state, action, reward)
            state = next_state
        
        save_to_video(frames, filename=save_video_name)

    avg_reward /= eval_episodes
    print(bd_stat.get_behavior_discriptor())

    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    # 
    return avg_reward

class Population_Trainer(object):

    def __init__(self, env, args):
        
        # base config
        self.args = args
        self.env = env
        self.population_size = args.population_size

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.results_dim = 1

        # training config
        self.total_iter = 100000000
        self.max_timesteps = self.env._max_episode_steps * 10 # max timesteps each iteration
        self.warmup_timesteps = [self.max_timesteps] * self.population_size # warm up timesteps
        self.batch_size = args.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_agents()
    
    def init_agents(self):
        kwargs = {
            "state_dim": self.state_dim, 
            "action_dim": self.action_dim,
            "max_action": float(self.env.action_space.high[0]),
            "device": self.device,
            "args": self.args
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
            rb = ReplayBuffer(self.state_dim, self.action_dim, self.results_dim, self.device)
            self.replay_buffer_pools.append(rb)
    
    def eval(self,):
        for agent_id in range(self.population_size):
            agent = self.agent_pools[agent_id]
            prefix = "_".join(agent.prefix.split("/"))
            save_video_name = f"{prefix}_{agent.agent_id}.avi"
            eval_policy(agent, self.env, seed=np.random.randint(100000),save_video_name=save_video_name)
    

    def get_diversity_info(self, iter_):
        behavior_descriptor = []
        for agent_id in range(self.population_size):
            sample_done = False
            while not sample_done:
                samples = self.replay_buffer_pools[agent_id].sample(self.batch_size)
                results, not_done = samples[-2], samples[-1]
                sample_done = (not_done.mean() < 1)
            results = results[(not_done==0).squeeze()].mean(0)
            behavior_descriptor.append(results)
        diversity_metric = DiversityMetric(behavior_descriptor)
        diversity_metric.run()

        if iter_ % 100 == 0:
            self.writer.add_scalar("diversity", diversity_metric.diversity, iter_)
            stat_tmp = torch.stack(behavior_descriptor, dim=0).detach()
            stat_tmp = torch.var(stat_tmp, dim=0)
            name = [f"action_dim_{idx}" for idx in range(stat_tmp.shape[0]-2)] + ["times", "score_rate"]
            for i in range(stat_tmp.shape[0]):
                self.writer.add_scalar(name[i], stat_tmp[i], iter_)

        return diversity_metric.behavior_descriptor
        

if __name__ == "__main__":
        
    args = build_flag()
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    env = gym.make(args.env)
    set_seed(env, args.seed)

    population_trainer = Population_Trainer(env, args)
    population_trainer.eval()