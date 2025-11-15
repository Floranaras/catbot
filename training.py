import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

def calculate_manhattan_distance(state):
    """Calculate Manhattan distance between agent and cat from state encoding."""
    agent_row = state // 1000
    agent_col = (state // 100) % 10
    cat_row = (state // 10) % 10
    cat_col = state % 10
    return abs(agent_row - cat_row) + abs(agent_col - cat_col)


#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    # Q-learning hyperparameters
    learning_rate = 0.15          # Alpha: how much new info overrides old info
    discount_factor = 0.97        # Gamma: importance of future rewards
    epsilon_start = 1.0           # Initial exploration rate
    epsilon_end = 0.01            # Minimum exploration rate
    epsilon_decay = 0.9996        # Decay rate for epsilon
    epsilon = epsilon_start
    
    max_steps_per_episode = 200   # Maximum steps before episode terminates

    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
               
        # Step 1: Reset the environment
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            # Step 2: Epsilon-greedy action selection
            if random.random() < epsilon:
                # Explore: choose random action
                action = env.action_space.sample()
            else:
                # Exploit: choose best action from Q-table
                action = int(np.argmax(q_table[state]))
            
            # Calculate distance before action
            prev_distance = calculate_manhattan_distance(state)
            
            # Step 3: Take action and observe next state
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Step 4: Compute reward manually
            # Reward structure:
            # - Large positive reward for catching the cat
            # - Small positive reward for moving closer
            # - Small negative reward for moving away
            # - Small negative reward for each step (encourages efficiency)
            
            if done:
                # Caught the cat - large positive reward
                reward = 100.0
            else:
                # Calculate new distance
                new_distance = calculate_manhattan_distance(next_state)
                
                # Reward for distance change
                if new_distance < prev_distance:
                    # Moving closer
                    reward = 10.0
                elif new_distance > prev_distance:
                    # Moving away
                    reward = -5.0
                else:
                    # Same distance
                    reward = -1.5
                
                # Small penalty for each step to encourage efficiency
                reward -= 0.5
            
            # Step 5: Update Q-table using Q-learning formula
            # Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])
            
            new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
            q_table[state][action] = new_value
            
            # Move to next state
            state = next_state
            steps += 1
        
        # Decay epsilon for less exploration over time
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        
        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table
