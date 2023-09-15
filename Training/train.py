from stable_baselines3 import A2C, DQN, PPO
import inspect
from CybORG import CybORG
from multi_agent import MultiAgent
import argparse
import os
import wandb

# Let user select the configurations
# Example command: python3 train.py --green --n_steps 27000 --algo PPO
parser = argparse.ArgumentParser()
parser.add_argument('--green', help="Pass this argument if the green agent should be included, False by default",
                    action='store_true')
parser.add_argument('--n_steps', help="Number of steps between PPO/A2C training", type=int, default=9000)
ALGOS = ["A2C", "DQN", "PPO"]
parser.add_argument("--algo", help="RL Algorithm", type=str.upper, required=True, choices=ALGOS)

args = parser.parse_args()
use_green = args.green
n_steps = args.n_steps # Allow for increase from hyperparam tuning to reduce variance
algo = args.algo

# Default configurations (not included in user input)
random_strategy = True
logging_path = log_path = os.path.join('Logs')

# Initialise environment
path = str(inspect.getfile(CybORG))
cyborg_path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
env = MultiAgent(config_path=cyborg_path, use_green=use_green, random_strategy=random_strategy)
n_timesteps = 4_000_000 # Increased to 4m for full training
wandb.login(key="7b4d820031986bf4003713b566df10bb1fb9f145")

# Training configuration with hyperparamaters identified in tuning for selected algorithm
if algo == 'A2C':
    config = {
        'policy_type': 'MlpPolicy',
        'ent_coef': 0.0036873574604549407, 
        'gae_lambda': 0.9219905738183752, 
        'gamma': 0.9148309819209979, 
        'learning_rate': 0.0028424659054832567, 
        'n_steps': n_steps, 
        'vf_coef': 0.5
    }
    run = wandb.init(
        project=f"{algo} Final Training 4M",
        config=config,
        sync_tensorboard=True,
        entity="melaniecmeijer"
    )

    model = A2C(config["policy_type"], env, learning_rate=config["learning_rate"], n_steps=config["n_steps"],
                gamma=config["gamma"], ent_coef=config["ent_coef"], gae_lambda=config["gae_lambda"],
                vf_coef=config["vf_coef"], verbose=0, tensorboard_log=logging_path)
    
elif algo == 'DQN':
    config = {
        'policy_type': 'MlpPolicy',
        'batch_size': 64,
        'buffer_size': 35000,
        'exploration_final_eps': 0.06132456749954254,
        'exploration_fraction': 0.4807709540619777,
        'gamma': 0.9404650858948974,
        'learning_rate': 0.0025458303568227842,
        'target_update_interval': 4000,
        'train_freq': 100
    }
    run = wandb.init(
        project=f"{algo} Final Training 4M",
        config=config,
        sync_tensorboard=True,
        entity="melaniecmeijer"
    ) 

    model = DQN(config["policy_type"], env, learning_rate=config["learning_rate"], batch_size=config["batch_size"], 
                buffer_size=config["buffer_size"], exploration_fraction=config["exploration_fraction"], 
                exploration_final_eps=config["exploration_final_eps"], gamma=config["gamma"], verbose=0,
                tensorboard_log=logging_path)

else: # PPO
    config = {
    'policy_type': 'MlpPolicy',
    'batch_size': 32, 
    'clip_range': 0.24871811066059937, 
    'ent_coef': 0.0024817668503458617, 
    'gae_lambda': 0.9558492466352743, 
    'gamma': 0.9617027667012086, 
    'learning_rate': 0.00023209024806257562, 
    'n_epochs': 6, 
    'n_steps': n_steps, 
    'vf_coef': 0.5
    }
    run = wandb.init(
        project=f"{algo} Final Training 4M",
        config=config,
        sync_tensorboard=True,
        entity="melaniecmeijer"
    )
    
    model = PPO(config["policy_type"], env, learning_rate=config["learning_rate"], n_steps=config["n_steps"],
            batch_size=config["batch_size"], gamma=config["gamma"], ent_coef=config["ent_coef"],
            clip_range=config["clip_range"], gae_lambda=config["gae_lambda"], n_epochs=config["n_epochs"],
            vf_coef=config["vf_coef"], verbose=1, tensorboard_log=logging_path)

# Train model
model.learn(total_timesteps=n_timesteps)
run.finish()

# Save model
model_path = os.path.join('Saved_Models', f"{algo}_final_model" )
model.save(os.path.join(model_path))