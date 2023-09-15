import optuna
import os
import matplotlib.pyplot as plt
from pprint import pprint

STUDY_PATH = os.path.join('Tuning', 'studies_ppo.db') # Select desired studies database

loaded_study = optuna.load_study(storage=f"sqlite:///{STUDY_PATH}", study_name="ppo-study") # Specify correct study_name

# Access tuning results
print("PPO best parameters:")
pprint(loaded_study.best_params)

# History plot
# optuna.visualization.matplotlib.plot_optimization_history(loaded_study)
# plt.title('A2C Optimization History Plot')
# plt.ylabel('Objective value (total reward)')

# Parameter importance plot
optuna.visualization.matplotlib.plot_param_importances(loaded_study)
plt.title('PPO Hyperparameters Importance')

plt.show()