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
from process_state import clip_reward, makeState, getFrame
import config
import os
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")
torch.autograd.set_detect_anomaly(True)

def create_experiment_dir(config_name: str) -> str:
    """Create a directory for the current experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join('experiments', f"{config_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def train(config_name: str = "default"):
    # Get configuration
    cfg = getattr(config, f"{config_name}_config", config.default_config)
    logger.info(f"Using configuration: {config_name}")
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(config_name)
    
    # Initialize environment
    env = gym.make(cfg.GAME_NAME, render_mode="rgb_array")
    logger.info(f"Environment created: {cfg.GAME_NAME}")
    
    # Initialize networks
    y = CNN.NeuralNetwork(env.action_space.n, None).to(device)
    target_y = CNN.NeuralNetwork(env.action_space.n, None).to(device)
    logger.info("Neural networks initialized")
    
    # Initialize training components
    loss_fn = nn.HuberLoss()
    optimizer = optim.Adam(y.parameters(), lr=cfg.LEARNING_RATE)
    agent = DQN.DQN(
        replay_memory_size=cfg.REPLAY_MEMORY_SIZE,
        batch_size=cfg.BATCH_SIZE,
        gamma=cfg.GAMMA,
        epsilon=cfg.EPSILON,
        epsilon_min=cfg.EPSILON_MIN,
        epsilon_decay=cfg.EPSILON_DECAY
    )
    
    # Initialize state buffer
    state = deque(maxlen=4)
    
    # Load pretrained model if specified
    if cfg.USE_PRETRAINED:
        logger.info(f"Loading pretrained model from {cfg.PRETRAINED_MODEL_PATH}")
        agent.loadModel(y, cfg.PRETRAINED_MODEL_PATH)
    
    # Initialize metrics
    frames_seen = 0
    rewards = []
    avgrewards = []
    loss = []
    episode_times = []
    
    # Initialize wandb if enabled
    if cfg.USE_WANDB:
        from wandb import wandb
        wandb.init(
            project=cfg.WANDB_PROJECT + "_" + cfg.GAME_NAME,
            entity=cfg.WANDB_ENTITY,
            config=vars(cfg),
            name=f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Training loop
    for episode in range(1, cfg.EPISODES + 1):
        episode_start_time = time.time()
        obs, _ = env.reset()  # New Gymnasium API returns (obs, info)
        cumureward = 0
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
            
            env.render()
            cache = state.copy()
            state.append(getFrame(obs))
            agent.update_replay_memory((makeState(cache), action, clip_reward(reward), makeState(state), done))
            
            # Train the agent
            if len(agent.replay_memory) >= cfg.START_TRAINING_AT_STEP and frames_seen % cfg.TRAINING_FREQUENCY == 0:
                loss.append(agent.train(y, target_y, loss_fn, optimizer))
            
            # Update target network
            if len(agent.replay_memory) >= cfg.START_TRAINING_AT_STEP and frames_seen % cfg.TARGET_NET_UPDATE_FREQUENCY == 0:
                target_y.load_state_dict(y.state_dict())
                logger.info("Target network updated")
            
            frames_seen += 1
            cumureward += reward
            
            # Save model periodically
            if frames_seen % cfg.SAVE_FREQUENCY == 0:
                agent.saveModel(y, f"{cfg.GAME_NAME}_{config_name}")
            
            if done:
                break
            
        # Calculate episode metrics
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        rewards.append(cumureward)
        avgrewards.append(np.mean(rewards[-100:]))  # Last 100 episodes average
        
        # Log metrics
        metrics = {
            "Reward per episode": cumureward,
            "Average reward (last 100)": avgrewards[-1],
            "Episode time": episode_time,
            "Average episode time": np.mean(episode_times[-100:]),
            "Epsilon": agent.EPSILON,
            "Frames seen": frames_seen,
            "Replay memory size": len(agent.replay_memory)
        }
        
        if loss:
            metrics["Loss"] = np.mean(loss)
            loss = []
        
        # Log to wandb if enabled
        if cfg.USE_WANDB:
            wandb.log(metrics)
        
        # Log episode summary
        logger.info(
            f"Frames_seen {frames_seen}| "
            f"Episode {episode}/{cfg.EPISODES} | "
            f"Score: {cumureward:.2f} | "
            f"Avg Reward: {avgrewards[-1]:.2f} | "
            f"Epsilon: {agent.EPSILON:.4f} | "
            f"Time: {episode_time:.2f}s"
        )
    
    # Close environment
    env.close()

if __name__ == "__main__":
    # Example usage with different configurations
    train("breakout")
    # train("space_invaders")
    # train("pong")
    # train("robotank")
    # train("vizdoom_defend")
    #train("vizdoom_deathmatch")
    # train("flappy_bird")