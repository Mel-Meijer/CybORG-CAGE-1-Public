import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.BlueLoadAgent import BlueLoadAgent
from CybORG.Agents.SimpleAgents.GreenAgent import GreenAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents.SimpleAgents.BlueRandomAgent import BlueRandomAgent

MAX_EPS = 10
agent_name = 'Blue'

def wrap(env):
    # return OpenAIGymWrapper(agent_name, EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(env))))
    return ChallengeWrapper(env=env, agent_name=agent_name)


if __name__ == "__main__":
    cyborg_version = '1.2'
    scenario = 'Scenario1b'
    name = "Melanie Meijer"
    team = "BSc project"
    name_of_agent = "Random Blue Agent without green"

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Change this line to load your agent / model file
    # agent = BlueLoadAgent('/Users/Melanie/repos/individual-project-cm3203/Training/Saved Models/DQN_final_model_rg.zip')
    agent = BlueRandomAgent()

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    # Specify output file name
    file_name = '/Users/Melanie/repos/individual-project-cm3203/Evaluation/Random_rng_eval.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{1.0}, {scenario}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:
            # Add green agent in here if desired; 'Green': GreenAgent
            cyborg = CybORG(path, 'sim', agents={'Red': red_agent}) 
            wrapped_cyborg = wrap(cyborg)
            wrapped_cyborg.set_seed(117) # add seed for reproducibility (same seed as og challenge)

            # observation = wrapped_cyborg.reset()
            observation = cyborg.reset().observation

            # action_space = wrapped_cyborg.get_action_space(agent_name)
            action_space = cyborg.get_action_space(agent_name)

            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):
                    action = agent.get_action(observation, action_space)

                    # observation, rew, done, info = wrapped_cyborg.step(action)
                    result = cyborg.step(agent_name, action)

                    # r.append(rew)
                    r.append(result.reward)

                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
                total_reward.append(sum(r))
                actions.append(a)

                observation = cyborg.reset().observation
                # observation = wrapped_cyborg.reset()
                
            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
