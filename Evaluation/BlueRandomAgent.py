import random
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import Monitor, Analyse, Misinform, Remove, Restore, Sleep


class BlueRandomAgent(BaseAgent):
    def __init__(self):
        self.action_space = [
                Monitor,
                Analyse,
                Misinform,
                Remove,
                Restore,
                Sleep
                ]
        self.hostnames = [
                'User0',
                'User1',
                'User2',
                'User3',
                'User4',
                'Enterprise0',
                'Enterprise1',
                'Enterprise2',
                'Defender',
                'Op_Host0', 
                'Op_Host1', 
                'Op_Host2', 
                'Op_Server0'
                ]

    def train(self, results: Results):
        # Not needed for this agent
        pass

    def get_action(self, observation, action_space):
        # Pick a random action
        action = random.choice(self.action_space)
        
        # Select random hostname
        hostname = random.choice(self.hostnames)
        
        # Assume a single session in the action space
        try:
            session = list(action_space['session'].keys())[0]
        except:
            session = 0

        if action == Sleep:
            return action()
        elif action == Monitor:
            # Does not have hostname as input parameter, all other actions do
            return action(agent='Blue', session=session)
        return action(agent='Blue', session=session, hostname=hostname)

    def end_episode(self):
        # Not needed for this agent
        pass

    def set_initial_values(self, action_space, observation):
        # Not needed for this agent
        pass