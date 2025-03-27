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
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast

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

# Performance optimizations
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    cudnn.benchmark = True  # Enable cuDNN autotuner
    cudnn.deterministic = False  # Disable deterministic mode
logger.info(f"Using {device} device")
torch.autograd.set_detect_anomaly(False)  # Disable anomaly detection

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
    
    # Initialize environment with optimized settings
    env = gym.make(cfg.GAME_NAME, render_mode="rgb_array")
    # Only record videos for evaluation episodes
    if cfg.RECORD_VIDEOS:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=f"{experiment_dir}/videos",
            episode_trigger=lambda episode_id: episode_id % cfg.VIDEO_RECORD_FREQUENCY == 0
        )
    logger.info(f"Environment created: {cfg.GAME_NAME}")
    
    # Initialize networks with performance optimizations
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
    
    # Initialize state buffer with numpy arrays for faster processing
    state = deque(maxlen=4)
    
    # Load pretrained model if specified
    if cfg.USE_PRETRAINED:
        logger.info(f"Loading pretrained model from {cfg.PRETRAINED_MODEL_PATH}")
        agent.loadModel(y, cfg.PRETRAINED_MODEL_PATH + f"{cfg.GAME_NAME}.pth")
    
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
    
    # Training loop with optimizations
    for episode in range(1, cfg.EPISODES + 1):
        episode_start_time = time.time()
        obs, _ = env.reset()
        cumureward = 0
        lives = cfg.LIVES
        state.clear()
        
        # Pre-allocate state buffer
        for _ in range(4):
            state.append(getFrame(obs))
        
        while True:
            # Use torch.no_grad() for inference
            with torch.no_grad():
                action = agent.getPrediction(makeState(state)/255, y)
            
            # Batch frame repeats for efficiency
            for _ in range(cfg.FRAME_REPEAT):
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if done or reward == 1:
                    break
            
            # Process frame and update state efficiently
            cache = state.copy()
            state.append(getFrame(obs))
            agent.update_replay_memory((makeState(cache), action, clip_reward(reward), makeState(state), done))
            
            # Train with mixed precision
            if len(agent.replay_memory) >= cfg.START_TRAINING_AT_STEP and frames_seen % cfg.TRAINING_FREQUENCY == 0:

                with autocast():
                    loss_value = agent.train(y, target_y, loss_fn, optimizer)
                    loss.append(loss_value)
            
            # Update target network
            if len(agent.replay_memory) >= cfg.START_TRAINING_AT_STEP and frames_seen % cfg.TARGET_NET_UPDATE_FREQUENCY == 0:
                target_y.load_state_dict(y.state_dict())
                logger.info("Target network updated")
            
            frames_seen += 1
            cumureward += reward
            
            # Save model periodically
            if frames_seen % cfg.SAVE_FREQUENCY == 0:
                agent.saveModel(y, cfg.PRETRAINED_MODEL_PATH + f"{cfg.GAME_NAME}.pth")
            
            if done:
                break
        
        # Calculate episode metrics
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        rewards.append(cumureward)
        avgrewards.append(np.mean(rewards[-100:]))
        
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
    # Set number of worker processes for data loading
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')
    
    # Example usage with different configurations
    train("breakout")
    # train("space_invaders")
    # train("pong")
    # train("robotank")
    # train("vizdoom_defend")
    #train("vizdoom_deathmatch")
    # train("flappy_bird")