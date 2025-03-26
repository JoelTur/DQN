from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # Training parameters
    EPISODES: int = 500000
    START_TRAINING_AT_STEP: int = 50000
    TRAINING_FREQUENCY: int = 4
    TARGET_NET_UPDATE_FREQUENCY: int = 10000
    
    # Model parameters
    LEARNING_RATE: float = 0.00025
    GAMMA: float = 0.99
    
    # Replay memory parameters
    REPLAY_MEMORY_SIZE: int = 2*10**5
    BATCH_SIZE: int = 32
    
    # Exploration parameters
    EPSILON: float = 1.0
    EPSILON_MIN: float = 0.1
    EPSILON_DECAY: float = 2*10**5
    
    # Environment parameters
    FRAME_REPEAT: int = 4  # Number of times to repeat each action (4 for FlappyBird/Doom, 1 for Atari)
    
    # Save frequency
    SAVE_FREQUENCY: int = 10000  # Save model every N frames
    
    # Game-specific settings
    GAME_NAME: str = "VizdoomDeathmatch-v0"
    LIVES: int = 0  # Number of lives (5 for Breakout, 3 for SpaceInvaders, 0 for Pong, 3 for Robotank)
    
    # Optional settings
    USE_PRETRAINED: bool = False
    PRETRAINED_MODEL_PATH: Optional[str] = f"models/"
    
    # Weights & Biases settings
    USE_WANDB: bool = True
    WANDB_PROJECT: str = "DQN"
    WANDB_ENTITY: str = "neuroori"

# Create default configuration
default_config = TrainingConfig()

# Game-specific configurations
breakout_config = TrainingConfig(
    GAME_NAME="BreakoutDeterministic-v4",
    LIVES=5,
    FRAME_REPEAT=1
)

space_invaders_config = TrainingConfig(
    GAME_NAME="SpaceInvadersDeterministic-v4",
    LIVES=3,
    FRAME_REPEAT=1
)

pong_config = TrainingConfig(
    GAME_NAME="PongDeterministic-v4",
    LIVES=0,
    FRAME_REPEAT=1
)

robotank_config = TrainingConfig(
    GAME_NAME="RobotankDeterministic-v4",
    LIVES=3,
    FRAME_REPEAT=1
)

vizdoom_defend_config = TrainingConfig(
    GAME_NAME="VizdoomDefendCenter-v0",
    FRAME_REPEAT=4
)

vizdoom_deathmatch_config = TrainingConfig(
    GAME_NAME="VizdoomDeathmatch-v0",
    FRAME_REPEAT=4
)

flappy_bird_config = TrainingConfig(
    GAME_NAME="FlappyBird-v0",
    FRAME_REPEAT=4
) 