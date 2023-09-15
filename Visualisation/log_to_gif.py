from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import matplotlib.patches as patches
import glob
import os
from PIL import Image
from natsort import natsorted


def log_to_dicts(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()[8:] # Skip config info
    results_dicts = []
    step_dict = {"hosts":[]} # Each dict represents one step

    for line in lines:
        # Read step and agent information
        if "Step number" in line:
            step_dict["step_number"] = int(line.lstrip('Step number: ').strip())
        elif "Blue" in line:
            step_dict["blue_action"] = line.lstrip('Blue action: ').strip()
        elif "Green" in line:
            step_dict["green_action"] = line.lstrip('Green action: ').strip()
        elif "Red" in line:
            step_dict["red_action"] = line.lstrip('Red action: ').strip()
        elif "Step reward" in line:
            step_dict["step_reward"] = float(line.lstrip('Step reward: ').strip())
        elif "Total" in line:
            step_dict["total_reward"] = float(line.lstrip('Total reward: ').strip())
        
        # Read all host status information
        elif "10.0" in line and "|" in line:
            row = line.split("|")
            
            # Gather info for each individual host
            host = {} 
            host["subnet"] = row[1].strip()
            host["ip_addr"] = row[2].strip()
            host["hostname"] = row[3].strip()
            host["known"] = "True" in row[4].strip()
            host["scanned"] = "True" in row[5].strip()
            host["access"] = row[6].strip()
                     
            # Append host info into step dictionary
            step_dict["hosts"].append(host)
        
        elif "End of step" in line:
            # Add step dictionary to results and reset for next step
            results_dicts.append(step_dict)
            step_dict = {"hosts":[]} 
            
        # Skip unnecessary lines (i.e. empty and table outlines)
        else:
            continue

    file.close()
    return results_dicts

def find_loc(hostname):
    # Retrieve the locations of each host on the image
    host_locations = {
        "Defender": (1365, 805),
        "Enterprise0": (1185, 158),
        "Enterprise1": (1371, 158),
        "Enterprise2": (1553, 158),
        "Op_Host0": (2314, 805),
        "Op_Host1": (2475, 805),
        "Op_Host2": (2634, 805),
        "Op_Server0": (2478, 158),
        "User0": (8, 814),
        "User1": (180, 814),
        "User2": (354, 814),
        "User3": (528, 814),
        "User4": (702, 814)
    }
    return host_locations[hostname]

def add_highlight(hostname, known, scanned, access):
    # Select corresponding colour and linestyle for host status
    if not known and access == "None":
        facecolor = (0,0,1,0.4)
        edgecolor = 'b'
        linestyle = '-'
    elif known and access == "None":
        facecolor = (0,0,1,0.4)
        edgecolor = 'r'
        if scanned:
            linestyle = '-'
        else:
            linestyle = '--'
    elif access == "User":
        facecolor = (1,0.5,0,0.4)
        edgecolor = 'r'
        linestyle = '--'
    else:
        facecolor = (1,0,0,0.4)
        edgecolor = 'r'
        linestyle = '-'

    # Select the correct size and location depending on host
    if "Enterprise" in hostname or hostname == "Op_Server0":
        width = 135
        height = 185
    else: # Regular, smaller host
        width = 154
        height = 148
    xy = find_loc(hostname)

    highlight_patch = patches.Rectangle(xy, width, height, facecolor=facecolor, edgecolor=edgecolor, linestyle=linestyle)
    return highlight_patch

if __name__ == "__main__":
    """
    Create a gif that shows the change in the state of the network over the course of the episode
        from the logs created in table_parser.py

    Input: log file created by run_episode.py
    Output: gif of network diagram changing with highlights at each node from actions taken
        played after one another sequentially
    """
    # Get a dictionary representation of the episode text file
    results_dicts = log_to_dicts("episode_results.txt")
    
    # Remove any previously generated images
    img_dir = 'images'
    for f in os.listdir(img_dir):
        os.remove(os.path.join(img_dir, f))

    # Save the image into the same folder and turn it into a matplotlib plot
    fig, ax = plt.subplots(figsize=(12, 7)) # Leave enough space for title line(s)
    base_img = pltimg.imread("labelled_network_diagram.png")
    ax.imshow(base_img)


    # Add title to plot
    plt.title("Initial Environment State", loc="left")
    # Include subnets and IP addresses in image as well (changes every episode)
    plt.text(x=180, y=545, s=results_dicts[1]["hosts"][8]["subnet"], fontsize="xx-small") # Subnet 1
    plt.text(x=1180, y=545, s=results_dicts[1]["hosts"][0]["subnet"], fontsize="xx-small") # Subnet 2
    plt.text(x=2295, y=550, s=results_dicts[1]["hosts"][4]["subnet"], fontsize="xx-small") # Subnet 3
    plt.axis('off')

    # Start off with initial environment state (not included in the results_dict)
    unknown_hosts = ["Defender", "Enterprise0", "Enterprise1", "Enterprise2", "Op_Host0", "Op_Host1", "Op_Host2",
        "Op_Server0", "User1", "User2", "User3", "User4"]
    for hostname in unknown_hosts:
        ax.add_patch(add_highlight(hostname, False, False, "None"))
    # Highlight initial red foothold
    ax.add_patch(add_highlight("User0", True, False, "Privileged"))
    plt.savefig("images/img0.png")
    plt.cla() # Clear plot

    # Create and save images for each consequent step from the results_dict
    for i, step_dict in enumerate(results_dicts):
        # Re-Create plot with base network diagram
        base_img = pltimg.imread("labelled_network_diagram.png")
        ax.imshow(base_img)

        # State agent actions and rewards
        plt.title(f"Step number {i+1}", loc="left")
        plt.text(x=900, y=-40, s = f"Blue Action: {step_dict['blue_action']}", fontsize="small")
        plt.text(x=900, y=0, s = f"Green Action: {step_dict['green_action']}", fontsize="small")
        plt.text(x=900, y=40, s = f"Red Action: {step_dict['red_action']}", fontsize="small")
        plt.text(x=1600, y=-40, s = f"Step Reward: {step_dict['step_reward']}", fontsize="small")
        plt.text(x=1600, y=0, s = f"Total Reward: {step_dict['total_reward']}", fontsize="small")
        
        # Add subnet IP ranges
        plt.text(x=180, y=545, s=results_dicts[1]["hosts"][8]["subnet"], fontsize="xx-small") # Subnet 1
        plt.text(x=1180, y=545, s=results_dicts[1]["hosts"][0]["subnet"], fontsize="xx-small") # Subnet 2
        plt.text(x=2295, y=550, s=results_dicts[1]["hosts"][4]["subnet"], fontsize="xx-small") # Subnet 3

        # Add corresponding highlight to each host
        for host in step_dict["hosts"]:
            ax.add_patch(add_highlight(host["hostname"], host["known"], host["scanned"], host["access"]))

        # Save image
        plt.axis('off')
        plt.savefig(f"images/img{i+1}.png")
        plt.cla() # Clear axes

    # Turn saved images into a gif
    episode_images = [Image.open(step_img) for step_img in natsorted(glob.glob("images/img*.png"))]
    frame_one = episode_images[0]
    frame_one.save("episode_gif.gif", format="GIF", append_images=episode_images,
               save_all=True, duration=400, loop=0)