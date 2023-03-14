## Quality-Similar Diversity via Population Based Reinforcement Learning

Official implementation for **Quality-Similar Diversity via Population Based Reinforcement Learning (ICLR2023)**. This is a TD3 version of QSD experimented on MuJoCo benchmark. 

### Examples

Examples for MuJoCo and Atari are in ./videos .

### To start
Install the mujoco environment.

Run the following command to install other requirements
```Bash
pip install -r requirements.txt
```

### Running 
see exp.sh


For Atari, we have tried to provide a comprehensive description of the Atari training in the Appendix. Aside from the standard DQN model, population-based PPO training runs, the QSD part (as like the MuJoCo code), we have additionally incorporated the distributed reinforcement learning framework mentioned in [1] to address the high variances of RL training in Atari environments. Due to the company’s policy, the Atari source code is not allowed to be disclosed yet.
[1] Ye, Deheng, et al. "Towards playing full moba games with deep reinforcement learning." Advances in Neural Information Processing Systems 33 (2020): 621-632.

### Citation
```latex
@inproceedings{
anonymous2023qualitysimilar,
title={Quality-Similar Diversity via Population Based Reinforcement Learning},
author={Anonymous},
booktitle={Submitted to The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=bLmSMXbqXr},
note={under review}
}
```

