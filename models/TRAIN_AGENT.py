import torch
from torch import nn
from torch._C import device
from torch import optim
import numpy as np
from collections import deque, Counter
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
import random
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

def evaluate_policy(y, agent, episodes, cfg, wandb, experiment_dir):
    logger.info("Evaluating the policy...")
    env = gym.make(cfg.GAME_NAME, render_mode="rgb_array")

    total_rewards = []
    state = deque(maxlen= 4)
    for epi in range(episodes):
        obs, _ = env.reset()
        for i in range(np.random.randint(0, 30)):
            obs, reward, terminated, truncated, info = env.step(np.random.choice([2,3]))
        for i in range(4):
            state.append(getFrame(obs, cfg.GAME_NAME))
        done = False
        steps = 0
        episode_reward = 0

        while not done:
            with torch.no_grad():
                action = agent.pred(makeState(state), y)

            frame_reward = 0
            for _ in range(cfg.FRAME_REPEAT):
                obs, reward, terminated, truncated, info = env.step(action)  # New Gymnasium API
                done = terminated or truncated
                frame_reward += reward
                if done:
                    break
            steps += 1
            state.append(getFrame(obs, cfg.GAME_NAME))
            episode_reward += frame_reward
            if steps > cfg.MAX_STEPS_EVAL:
                done = True
        total_rewards.append(episode_reward)
        logger.info(f"Episode {(epi+1)}/{episodes} | " f"Steps {steps}/{5000} | " f"Reward {episode_reward} | ")
    
    env.close()
    avg_eval_score = sum(total_rewards) / len(total_rewards)
    logger.info(f"Done evaluating the policy with avg reward {avg_eval_score}!")
    wandb.log({"Avg eval Reward": avg_eval_score})
    return avg_eval_score

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
            video_folder=f"{experiment_dir}/videos_train",
            episode_trigger=lambda episode_id: episode_id % cfg.VIDEO_RECORD_FREQUENCY == 0
        )
    logger.info(f"Environment created: {cfg.GAME_NAME}")
    
    # Initialize networks with performance optimizations
    y = CNN.NeuralNetwork(env.action_space.n, None).to(device)
    target_y = CNN.NeuralNetwork(env.action_space.n, None).to(device)
    

    logger.info("Neural networks initialized")
    
    # Initialize training components
    loss_fn = nn.HuberLoss()
    optimizer = optim.Adam(y.parameters(), lr=cfg.LEARNING_RATE, eps=1e-3)

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
    timestep = 0
    param_updates = 0
    rewards = deque(maxlen=100)
    loss = []
    episode_times = deque(maxlen=100)
    batch_Q_values_max = None
    batch_Q_values_mean = None
    best_evaluation_score = 0
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

        # Randomly choose 2-3 actions to take before training
        #for i in range(np.random.randint(0, 30)):
        #    obs, reward, terminated, truncated, info = env.step(np.random.choice([2,3]))

                # Pre-allocate state buffer
        for _ in range(4):
            state.append(getFrame(obs, cfg.GAME_NAME))
        
        while True:
            # Use torch.no_grad() for inference

            with torch.no_grad():
                action = agent.epsilon_greedy(makeState(state), y)
            

            # Batch frame repeats for efficiency
            frame_reward = 0
            for _ in range(cfg.FRAME_REPEAT):
                obs, reward, terminated, truncated, info = env.step(action)
                frame_reward += reward
                done = terminated or truncated
                if done:
                    break
            
            # Process frame and update state efficiently
            cache = state.copy()
            state.append(getFrame(obs, cfg.GAME_NAME))
            agent.update_replay_memory(makeState(cache), makeState(state), action, clip_reward(frame_reward), done)
            
            # Train with mixed precision
            if agent.replay_memory.size() >= cfg.START_TRAINING_AT_STEP and timestep % cfg.TRAINING_FREQUENCY == 0:
                loss_value, batch_Q_values_max, batch_Q_values_mean = agent.train(y, target_y, loss_fn, optimizer)
                loss.append(loss_value)
                batch_Q_values_max = batch_Q_values_max
                batch_Q_values_mean = batch_Q_values_mean
            

            # Update target network
            if agent.replay_memory.size() >= cfg.START_TRAINING_AT_STEP and timestep % cfg.TARGET_NET_UPDATE_FREQUENCY == 0:
                target_y.load_state_dict(y.state_dict())
                logger.info("Target network updated")
            
            timestep += 1
            cumureward += frame_reward
            if timestep >= cfg.START_TRAINING_AT_STEP:
                agent.reduce_epsilon()
            # Save model periodically
            if timestep % cfg.EVAL_FREQUENCY == 0 and timestep > 5*10**6:
                evaluation_score = evaluate_policy(y, agent, 10, cfg, wandb, experiment_dir) 
                if evaluation_score > best_evaluation_score:
                    best_evaluation_score = evaluation_score
                    agent.saveModel(y, cfg.PRETRAINED_MODEL_PATH + f"{cfg.GAME_NAME}.pth")
            if timestep == cfg.PHASE_2_BEGINS_TIMESTEPS:
                prev_lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = cfg.LEARNING_RATE_PHASE_2
                cur_lr = optimizer.param_groups[0]['lr']
                logger.info(f"learning rate reduced from {prev_lr} to {cur_lr}")
            if done:
                break
        
        # Calculate episode metrics
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        rewards.append(cumureward)
        
        grad_norms = []
        if y.parameters() is not None:
            for p in y.parameters():
                if p.grad is not None:
                    grad_norms.append(p.grad.norm().item())
        
        # Log metrics
        metrics = {
            "Reward per episode": cumureward,
            "Average reward (last 100)": np.mean(rewards),
            "Episode time": episode_time,
            "Average episode time (last 100)": np.mean(episode_times),
            "Epsilon": agent.EPSILON,
            "Timestep": timestep,
            "Replay memory size": agent.replay_memory.size(),
            "Gradient norms": np.mean(grad_norms),
            "Batch Q values max": batch_Q_values_max,
            "Batch Q values mean": batch_Q_values_mean
        }
        
        if loss:
            metrics["Loss"] = loss[-1]
            loss = []
        
        # Log to wandb if enabled
        if cfg.USE_WANDB:
            try:
                wandb.log(metrics)
            except Exception as e:
                logger.error(f"Error logging to wandb: {e}")
        
        # Log episode summary
        logger.info(
            f"Timestep {timestep}| "
            f"Episode {episode}/{cfg.EPISODES} | "
            f"Score: {cumureward:.2f} | "
            f"Avg training Reward: {np.mean(rewards):.2f} | "
            f"Epsilon: {agent.EPSILON:.4f} | "
            f"Time: {episode_time:.2f}s | "
            f"Grad Norms: {np.mean(grad_norms):.2f} | "
            f"Batch Q values max: {batch_Q_values_max} | "
            f"Batch Q values mean: {batch_Q_values_mean} | "
            f"Best Avg eval Reward: {best_evaluation_score}")
    
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