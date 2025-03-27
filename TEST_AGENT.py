import torch
from torch import nn
from torch._C import device
from torch import optim
import numpy as np
from collections import deque
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import DQN
import CNN
import config
from process_state import getFrame, clip_reward, makeState
import time
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")
torch.autograd.set_detect_anomaly(True)

def test(config_name: str = "default", num_games: int = 100):
    # Get configuration
    cfg = getattr(config, f"{config_name}_config", config.default_config)
    logger.info(f"Using configuration: {config_name}")
    
    # Initialize environment
    env = gym.make(cfg.GAME_NAME, render_mode="human")
    logger.info(f"Environment created: {cfg.GAME_NAME}")
    
    # Initialize network
    y = CNN.NeuralNetwork(env.action_space.n, None).to(device)
    agent = DQN.DQN(
        replay_memory_size=cfg.REPLAY_MEMORY_SIZE,
        batch_size=cfg.BATCH_SIZE,
        gamma=cfg.GAMMA,
        epsilon=0,  # No exploration during testing
        epsilon_min=0,
        epsilon_decay=cfg.EPSILON_DECAY
    )
    
    # Load the model
    model_path = os.path.join('models', f"{cfg.GAME_NAME}.pth")
    agent.loadModel(y, model_path)

    
    # Initialize state buffer
    state = deque(maxlen=4)
    
    # Initialize metrics
    rewards = []
    avgrewards = []
    episode_times = []
    
    # Initialize wandb if enabled
    if cfg.USE_WANDB:
        from wandb import wandb
        wandb.init(
            project=cfg.WANDB_PROJECT + "_" + cfg.GAME_NAME + "_TEST",
            entity=cfg.WANDB_ENTITY,
            config=vars(cfg),
            name=f"{config_name}_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Testing loop
    for game in range(1, num_games + 1):
        episode_start_time = time.time()
        obs, _ = env.reset()  # New Gymnasium API returns (obs, info)
        score = 0
        lives = cfg.LIVES
        state.clear()
        
        # Initialize state buffer
        for _ in range(4):
            state.append(getFrame(obs))
        
        while True:
            action = agent.getPrediction(makeState(state)/255, y)
            
            # Repeat action based on configuration
            for _ in range(cfg.FRAME_REPEAT):
                obs, reward, terminated, truncated, info = env.step(action)  # New Gymnasium API
                done = terminated or truncated
                if done or reward == 1:
                    break
            
            state.append(getFrame(obs))
            score += reward
            env.render()
            
            if done and lives == 0:
                break
        
        # Calculate episode metrics
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        rewards.append(score)
        avgrewards.append(np.mean(rewards[-100:]))  # Last 100 episodes average
        
        # Log metrics
        metrics = {
            "Score": score,
            "Average score (last 100)": avgrewards[-1],
            "Episode time": episode_time,
            "Average episode time": np.mean(episode_times[-100:]),
            "Game": game
        }
        
        # Log to wandb if enabled
        if cfg.USE_WANDB:
            wandb.log(metrics)
        
        # Log episode summary
        logger.info(
            f"Game {game}/{num_games} | "
            f"Score: {score:.2f} | "
            f"Avg Score: {avgrewards[-1]:.2f} | "
            f"Time: {episode_time:.2f}s"
        )
    
    # Log final results
    final_avg_score = np.mean(rewards)
    logger.info(f"Testing completed. Final average score: {final_avg_score:.2f}")
    
    # Log final score to wandb if enabled
    if cfg.USE_WANDB:
        wandb.log({"Final average score": final_avg_score})
    
    # Close environment
    env.close()

if __name__ == "__main__":
    # Example usage with different configurations
    test("breakout", num_games=100)
    # test("space_invaders", num_games=100)
    # test("pong", num_games=100)
    # test("robotank", num_games=100)
    # test("vizdoom_defend", num_games=100)
    # test("vizdoom_deathmatch", num_games=100)
    # test("flappy_bird", num_games=100)