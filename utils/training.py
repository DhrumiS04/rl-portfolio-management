import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def train_reinforce(env, agent, num_episodes, eval_interval=20, eval_episodes=5, save_path=None):
    """
    Train a REINFORCE agent.
    
    Args:
        env: The environment.
        agent: The REINFORCE agent.
        num_episodes (int): Number of episodes to train for.
        eval_interval (int): Interval for evaluation during training.
        eval_episodes (int): Number of episodes for evaluation.
        save_path (str): Path to save the trained agent.
        
    Returns:
        tuple: (agent, training_rewards, evaluation_rewards)
    """
    training_rewards = []
    evaluation_rewards = []
    
    for episode in tqdm(range(num_episodes)):
        # Reset the environment and agent episode history
        state, _ = env.reset()
        agent.reset_episode()
        done = False
        truncated = False
        episode_reward = 0
        
        # Run an episode
        while not done and not truncated:
            # Select action
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store reward
            agent.store_reward(reward)
            episode_reward += reward
            
            # Update state
            state = next_state
        
        # Update agent after episode
        agent.update()
        
        # Record training reward
        training_rewards.append(episode_reward)
        
        # Evaluate the agent periodically
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent, eval_episodes)
            evaluation_rewards.append(eval_reward)
            
            print(f"\nEpisode {episode+1}/{num_episodes}")
            print(f"Training reward: {episode_reward:.4f}")
            print(f"Evaluation reward: {eval_reward:.4f}")
            
            # Save the best agent
            if save_path and (len(evaluation_rewards) == 1 or eval_reward > max(evaluation_rewards[:-1])):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                agent.save(save_path)
                print(f"Saved model to {save_path}")
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.plot(training_rewards, label='Training Rewards')
    plt.plot(np.arange(eval_interval-1, num_episodes, eval_interval), evaluation_rewards, label='Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return agent, training_rewards, evaluation_rewards


def train_a2c(env, agent, num_episodes, eval_interval=20, eval_episodes=5, save_path=None):
    """
    Train an A2C agent.
    
    Args:
        env: The environment.
        agent: The A2C agent.
        num_episodes (int): Number of episodes to train for.
        eval_interval (int): Interval for evaluation during training.
        eval_episodes (int): Number of episodes for evaluation.
        save_path (str): Path to save the trained agent.
        
    Returns:
        tuple: (agent, training_rewards, evaluation_rewards)
    """
    training_rewards = []
    evaluation_rewards = []
    
    for episode in tqdm(range(num_episodes)):
        # Reset the environment and agent episode history
        state, _ = env.reset()
        agent.reset_episode()
        done = False
        truncated = False
        episode_reward = 0
        
        # Run an episode
        while not done and not truncated:
            # Select action
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store reward
            agent.store_reward(reward)
            episode_reward += reward
            
            # Update state
            state = next_state
        
        # Update agent after episode
        total_loss, actor_loss, critic_loss, entropy_loss = agent.update()
        
        # Record training reward
        training_rewards.append(episode_reward)
        
        # Evaluate the agent periodically
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent, eval_episodes)
            evaluation_rewards.append(eval_reward)
            
            print(f"\nEpisode {episode+1}/{num_episodes}")
            print(f"Training reward: {episode_reward:.4f}")
            print(f"Evaluation reward: {eval_reward:.4f}")
            print(f"Actor loss: {actor_loss:.4f}, Critic loss: {critic_loss:.4f}, Entropy loss: {entropy_loss:.4f}")
            
            # Save the best agent
            if save_path and (len(evaluation_rewards) == 1 or eval_reward > max(evaluation_rewards[:-1])):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                agent.save(save_path)
                print(f"Saved model to {save_path}")
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.plot(training_rewards, label='Training Rewards')
    plt.plot(np.arange(eval_interval-1, num_episodes, eval_interval), evaluation_rewards, label='Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return agent, training_rewards, evaluation_rewards


def evaluate_agent(env, agent, num_episodes=5):
    """
    Evaluate an agent on multiple episodes.
    
    Args:
        env: The environment.
        agent: The agent.
        num_episodes (int): Number of episodes to evaluate on.
        
    Returns:
        float: Average reward over episodes.
    """
    total_reward = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not done and not truncated:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / num_episodes
