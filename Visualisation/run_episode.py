from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, RedMeanderAgent, BlueRandomAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table
from stable_baselines3 import PPO, A2C, DQN
import inspect
import argparse

def map_int_to_action(action_int):
    """
    Map actions to action ints within the Discret(53) action space:
    A global Sleep (0)
    A global Monitor (1)
    13 Analyse (one for each host) (2-14)
    13 Remove (one for each host) (15-27)
    13 Restore (one for each host) (28-40)
    13 Misinform (one for each host) (41-53)

    Hosts (ints 0-12 in this order):
    Defender, Enterprise0, Enterprise1, Enterprise2, Op_Host0, Op_Host1, Op_Host2, Op_Server, 
        User0, User1, User2, User3, User4
    """
    # base cases
    if action_int == 0:
        return "Sleep"
    elif action_int == 1:
        return "Monitor"
    
    # actions with individual hosts
    else:
        if action_int >= 2 and action_int <15:
            return "Analyse " + get_host_name(action_int-2)
        elif action_int >=15 and action_int <28:
            return "Remove " + get_host_name(action_int-15)
        elif action_int >=28 and action_int <41:
            return "Restore " + get_host_name(action_int-28)
        else:
            return "Misinform " + get_host_name(action_int-41)

# Implement python switch equivalent to translate host name to integer
def get_host_name(host_int):
    hosts={
            0: 'Defender',
            1: 'Enterprise0',
            2: 'Enterprise1',
            3: 'Enterprise2',
            4: 'Op_Host0',
            5: 'Op_Host1',
            6: 'Op_Host2',
            7: 'Op_Server',
            8: 'User0',
            9: 'User1',
            10: 'User2',
            11: 'User3',
            12: 'User4'}
    return hosts.get(host_int,"Invalid host")

if __name__ == "__main__":
    """
    Let user select all agents and time limit of episode through argparse
    Have their specified blue agent complete one episode (max steps) against specified env agents
    Output: text file with truth tables, agent actions and rewards after each step
    """
    # Save all text file output in list to write at the end of the episode
    lines = []

    # Get user input for episode configurations
    # Example command: python3 run_episode.py --model_file ../Training/Saved_Models/PPO_final_model_rg.zip --algo PPO --red b_line --green --max_steps 100
    ALGO_OPTIONS = ["A2C", "DQN", "PPO"]
    algos_dict = {"A2C":A2C, "DQN":DQN, "PPO":PPO, "NONE":None}

    RED_OPTIONS = ["b_line", "meander"]
    reds_dict = {"b_line":B_lineAgent, "meander":RedMeanderAgent}

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", help="Path to the saved model file, Random agent will be used if not provided",
                        type=str)
    parser.add_argument("--algo", help="RL Algorithm used in the saved model file", type=str.upper, choices=ALGO_OPTIONS, default="NONE")
    parser.add_argument('--red', help="Specify the red agent you'd like to use for this episode, meander by default",
                        choices=RED_OPTIONS, type=str.lower, default='meander')
    parser.add_argument('--green', help="Pass this argument if the green agent should be included, False by default",
                        action='store_true')
    parser.add_argument('--max_steps', help="Maximum number of steps for the episode, 100 by default", type=int, default=30)
    parser.add_argument('--log_name', help="Choose the filename for output log file, episode_results by default",
                        default='episode_results.txt')

    args = parser.parse_args()
    model_file = args.model_file
    algo = algos_dict[args.algo]
    red_agent = reds_dict[args.red]
    use_green = args.green
    max_steps = args.max_steps
    log_name = args.log_name

    # Save episode configurations into the results log file
    print("Logging to file " + str(log_name))
    lines.append("Episode Config Info")
    if model_file is not None:
        lines.append("Blue model used: "+ str(model_file))
        lines.append("Algorithm used: "+ str(args.algo))
    else:
        lines.append("No model file and/or algorithm specified")
        lines.append("Blue model used: Random Agent")

    lines.append("Red agent used: "+ str(args.red.capitalize()))
    lines.append("Green agent used? " + str(use_green))
    lines.append("Episode max steps limit: " + str(max_steps))
    lines.append(80*"-"+"\n")


    # Create environment
    # Get path to a .yaml scenario file which defines the network layout and agent action spaces
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    agents = {'Red': red_agent, 'Green': GreenAgent} if use_green else {'Red': red_agent}

    env = CybORG(path,'sim',agents=agents)
    env.set_seed(100) # Add seed for reproducibility

    # Load agent if user has specified a model file through argparse
    if model_file is not None and algo is not None:
        wrapped_env = ChallengeWrapper(env=env,agent_name='Blue')
        model = algo.load(model_file)

        obs = wrapped_env.reset()
        total_rew = 0

        for step in range(max_steps):
            action, _states = model.predict(obs)
            obs, rew, done, info = wrapped_env.step(action)
            total_rew += rew
            # Save step info
            lines.append("Step number: " + str(step+1))
            lines.append('Blue action: '+ map_int_to_action(action))

            red_action = wrapped_env.get_last_action('Green')
            lines.append('Green action: '+ str(red_action))

            red_action = wrapped_env.get_last_action('Red')
            lines.append('Red action: '+ str(red_action))

            lines.append('Step reward: '+ str(rew))
            lines.append('Total reward: '+ str(total_rew))

            true_state = env.get_agent_state('True')
            true_table = true_obs_to_table(true_state,env)
            lines.append(str(true_table))
            lines.append("End of step " + str(step+1))
            lines.append("")

    # Else use random agent
    else:
        agent = BlueRandomAgent() # Selects random action at each step
        
        results = env.reset(agent='Blue')
        obs = results.observation
        action_space = results.action_space
        
        total_rew = 0

        for step in range(max_steps):
            action = agent.get_action(obs,action_space=action_space)
            results = env.step(agent='Blue',action=action)
            obs = results.observation
            rew = results.reward
            total_rew += rew

            # Save step info
            lines.append("Step number: " + str(step+1))
            lines.append('Blue action: '+ str(action))
            
            red_action = env.get_last_action('Green')
            lines.append('Green action: '+ str(red_action))

            red_action = env.get_last_action('Red')
            lines.append('Red action: '+ str(red_action))

            lines.append('Step reward: '+ str(rew))
            lines.append('Total reward: '+ str(total_rew))
            
            true_state = env.get_agent_state('True')
            true_table = true_obs_to_table(true_state,env)
            lines.append(str(true_table))
            lines.append("End of step " + str(step+1))
            lines.append("")

    # Write output (stored in list lines) to text file
    with open(log_name, 'w') as file:
        file.write('\n'.join(lines))
    file.close()
