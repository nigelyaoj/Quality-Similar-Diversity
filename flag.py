import argparse
import torch

def build_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  
    parser.add_argument("--env", default="Hopper-v2")       
    parser.add_argument("--seed", default=0, type=int)             
    parser.add_argument("--start_timesteps", default=5e3, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)       
    parser.add_argument("--max_timesteps", default=1e6, type=int)  
    parser.add_argument("--expl_noise", default=0.1)                
    parser.add_argument("--batch_size", default=2048, type=int)      
    parser.add_argument("--discount", default=0.99)  
    parser.add_argument("--res_discount", default=1)      
    parser.add_argument("--tau", default=0.05)                     
    parser.add_argument("--policy_noise", default=0.1)              
    parser.add_argument("--noise_clip", default=0.5)               
    parser.add_argument("--policy_freq", default=2, type=int)      
    parser.add_argument("--save_freq", default=500, type=int)  
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", action="store_true") 
    parser.add_argument("--population_size", default=4, type=int)  
    parser.add_argument("--log_path", default="logs")
    parser.add_argument("--loss_weight_guide", default=0, type=float)
    parser.add_argument("--loss_weight_lambda", default=1, type=float)    
    parser.add_argument("--exp_name", default="debug2") 
    parser.add_argument("--device", default="cuda:0") 
    parser.add_argument("--lr", default=1e-3, type=float) 
    parser.add_argument("--reward_threshold_to_guide", default=0.8, type=float)
    parser.add_argument("--traj_len", default=5, type=int)
    parser.add_argument("--lstm_hidden_dim", default=256, type=int)
    parser.add_argument("--div_metric", default="MSE", type=str)
    

    args = parser.parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    return args