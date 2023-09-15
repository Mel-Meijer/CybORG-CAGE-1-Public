from CybORG.Agents import B_lineAgent, RedMeanderAgent, GreenAgent, BaseAgent
from CybORG import CybORG
# from time_limit import TimeLimit
from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit
from CybORG.Agents.Wrappers import ChallengeWrapper
from stable_baselines3.common.monitor import Monitor
import random

class MultiAgent(Wrapper):

    def __init__(self, config_path: str, use_green: bool = True, random_strategy: bool = True):
        self._config_path = config_path
        self._use_green = use_green
        self.random_strategy = random_strategy
        
        # Initialise environment
        cyborg = CybORG(self._config_path, 'sim', agents={'Red': B_lineAgent})
        env = ChallengeWrapper(env=cyborg, agent_name='Blue')
        super().__init__(env)

        self._b_line = self._create_env(B_lineAgent)
        self._meander = self._create_env(RedMeanderAgent)
        # Start with b_line red agent
        self.env = self._b_line

    def _create_env(self, red_agent: BaseAgent):
        agents = {'Red': red_agent, 'Green': GreenAgent} if self._use_green else {'Red': red_agent}
        cyborg = CybORG(self._config_path, 'sim', agents=agents)
        env = ChallengeWrapper(env=cyborg, agent_name='Blue')
        env = TimeLimit(env, max_episode_steps=100)
        return Monitor(env)

    def _random_env(self):
        r = random.randint(0, 1)
        self.env = self._b_line if r == 0 else self._meander

    def _switch_env(self):
        self.env = self._b_line if self.env == self._meander else self._meander

    def step(self, action):
        observation, reward, truncated, info = self.env.step(action)
        return observation, reward, truncated, info

    def reset(self, **kwargs):
        # Either switch randomly or alternate consistently depending on user input
        if self.random_strategy:
            self._random_env()
        else:
            self._switch_env()
        return self.env.reset(**kwargs)