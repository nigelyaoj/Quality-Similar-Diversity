# To run exp

ENV_1='Hopper-v2'
nohup python main.py --env ${ENV_1} --exp_name Hopper --population_size 8 --save_model \
--loss_weight_lambda 5 --loss_weight_guide 0 > Hopper_5_0.log &

ENV_1='Walker2d-v2'
nohup python main.py --env ${ENV_1} --exp_name Walker2d --population_size 8 --save_model \
--loss_weight_lambda 0.5 --loss_weight_guide 0 > Walker2d_05_0.log &

ENV_1='Humanoid-v2'
nohup python main.py --env ${ENV_1} --exp_name Humanoid --population_size 8 --save_model \
--loss_weight_lambda 10 --loss_weight_guide 1 > Humanoid_10_1.log &

# To evaluate

ENV_1='Hopper-v2'
nohup python eval_diversity.py --env ${ENV_1} --exp_name Hopper --population_size 8 --save_model \
--loss_weight_lambda 5 --loss_weight_guide 0 > eval_Hopper_5_0.log &

ENV_1='Walker2d-v2'
nohup python eval_diversity.py --env ${ENV_1} --exp_name Walker2d --population_size 8 --save_model \
--loss_weight_lambda 0.5 --loss_weight_guide 0 > eval_Walker2d_05_0.log &

ENV_1='Humanoid-v2'
nohup python eval_diversity.py --env ${ENV_1} --exp_name Humanoid --population_size 8 --save_model \
--loss_weight_lambda 10 --loss_weight_guide 1 > eval_Humanoid_10_1.log &