import os
import argparse
import optuna
import inspect
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from optuna.study import MaxTrialsCallback
from optuna.samplers import RandomSampler
from optuna.trial import TrialState

# Let user select which RL algorithm to use
# Example command: python3 tuning_ALL_optuna.py --algo ppo
parser = argparse.ArgumentParser()
ALGOS = ["a2c", "dqn", "ppo"]
parser.add_argument("--algo", help="RL Algorithm (a2c, ppo or dqn)", type=str.lower, required=True, choices=ALGOS)
args = parser.parse_args()
algo = args['algo']

STUDY_NAME = f"{algo}-study"
STUDY_PATH = os.path.join('Tuning', 'studies.db')

# Pruning and evaluation variables
N_TIMESTEPS = 2_000_000
N_WARMPUP_STEPS = N_TIMESTEPS // 3
N_INTERVAL_STEPS = 100_000
N_TRIALS = 50
N_STARTUP_TRIALS = 5
EVAL_FREQ = 40_000

# Initialise the environment
path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

# Include the GreenAgent
agents = {
    'Red': B_lineAgent,
    'Green': GreenAgent
}

cyborg = CybORG(path,'sim',agents=agents)
env = ChallengeWrapper(env=cyborg,agent_name='Blue')
env = TimeLimit(env, max_episode_steps=100)
env = Monitor(env)


# Create callback to incorporate pruner in evaluation
"""
Based on following source code:
Title: SB3_Simple Python Script
Author: Kento Nozawa
Date: February 1, 2023
Code version: Version 9
Availability: https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py
"""
class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if needed.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


# Define the objective function for tuning
def objective(trial):
    # Common hyperparameters for all three algorithms
    gamma = trial.suggest_loguniform('gamma', 0.9, 0.9997)
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-6, 0.003)

    # Define the hyperparameters to be tuned depending on selected algorithm
    if algo == 'a2c':
        ent_coef = trial.suggest_loguniform('ent_coef', 0.00001, 0.01)
        gae_lambda = trial.suggest_uniform('gae_lambda', 0.9, 1)
        vf_coef = trial.suggest_categorical('vf_coef', [0.5, 1])
        n_steps = trial.suggest_categorical('n_steps', list(range(1000, 11000, 1000)))
        
        model = A2C("MlpPolicy", env, learning_rate=learning_rate, n_steps=n_steps, gamma=gamma,
                ent_coef=ent_coef, gae_lambda=gae_lambda, vf_coef=vf_coef, verbose=0)

    elif algo == 'dqn':
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        buffer_size = trial.suggest_categorical('buffer_size', list(range(100000, 1100000, 100000)))
        exploration_fraction = trial.suggest_float('exploration_fraction', 0.1, 0.9)
        exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.05, 0.15)

        model = DQN("MlpPolicy", env, learning_rate=learning_rate, batch_size=batch_size, buffer_size=buffer_size,
                exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps, gamma=gamma,
                verbose=0)
        
    else:
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        ent_coef = trial.suggest_loguniform('ent_coef', 0.00001, 0.01)
        gae_lambda = trial.suggest_uniform('gae_lambda', 0.9, 1)
        n_epochs = trial.suggest_int('n_epochs', 3, 10)
        vf_coef = trial.suggest_categorical('vf_coef', [0.5, 1])
        n_steps = trial.suggest_categorical('n_steps', list(range(1000, 11000, 1000)))
        clip_range = trial.suggest_uniform('clip_range', 0.1, 0.4)

        model = PPO("MlpPolicy", env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, gamma=gamma,
                    ent_coef=ent_coef, clip_range=clip_range, gae_lambda=gae_lambda, n_epochs=n_epochs, vf_coef=vf_coef,
                    verbose=0)

  
    # Train the model using evaluation callback
    eval_callback = TrialEvalCallback(env, trial, eval_freq=EVAL_FREQ, deterministic=True, n_eval_episodes=50)
    model.learn(total_timesteps=N_TIMESTEPS, callback=eval_callback)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1000, deterministic=True)
    return mean_reward

# Set up the Optuna study with the MedianPruner to increase efficiency
# Do not prune before 1/3 of the timesteps have been completed, to avoid pruning high exploration phase
pruner = optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_WARMPUP_STEPS, 
                                     interval_steps=N_INTERVAL_STEPS)
study = optuna.create_study(direction='maximize', sampler=RandomSampler(), pruner=pruner,
                            storage=f"sqlite:///{STUDY_PATH}", load_if_exists=True, study_name=STUDY_NAME)

# Optimize the hyperparameters
# Use MaxTrialsCallback to ensure how many times trials will be performed across all processes
study.optimize(objective, callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE,))])