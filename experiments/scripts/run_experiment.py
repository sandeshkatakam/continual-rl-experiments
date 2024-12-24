import hydra
from omegaconf import DictConfig
import wandb
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path

from continual_rl.utils.logging import setup_logging
from continual_rl.agents import get_agent

@hydra.main(config_path="../../src/continual_rl/config", config_name="default_config")
def main(cfg: DictConfig):
    # Setup logging
    logger = setup_logging(cfg)
    
    # Set random seeds
    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)
    
    # Initialize WandB
    wandb.init(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        config=dict(cfg),
        name=cfg.experiment.name
    )
    
    # Create environment
    env = gym.make(cfg.env.name)
    
    # Initialize agent
    agent = get_agent(cfg)
    
    # Training loop
    for task_id in range(cfg.experiment.num_tasks):
        for episode in range(cfg.experiment.episodes_per_task):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(cfg.env.max_steps):
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                agent.update({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                })
                
                episode_reward += reward
                state = next_state
                
                if done or truncated:
                    break
            
            # Logging
            if episode % cfg.logging.log_interval == 0:
                wandb.log({
                    'task_id': task_id,
                    'episode': episode,
                    'reward': episode_reward
                })
            
            # Save checkpoints
            if episode % cfg.logging.save_interval == 0:
                save_path = Path(f"checkpoints/task_{task_id}_episode_{episode}.pt")
                agent.save(save_path)
    
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()