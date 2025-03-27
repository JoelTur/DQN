import skimage
import numpy as np
import config
from typing import Tuple, List

def clip_reward(reward: float) -> float:
    """Clip rewards to {-1, 0, 1} to help with training stability."""
    if reward > 0:
        return 1
    elif reward < 0:
        return -1
    else:
        return 0

def getFrame(x: np.ndarray, game_name: str = None) -> np.ndarray:
    """
    Process a frame from the game environment.
    
    Args:
        x: Raw frame from the environment
        game_name: Name of the game to determine proper cropping
        
    Returns:
        Processed frame as a grayscale image
    """
    # Define crop regions for different games
    crop_regions = {
        "BreakoutDeterministic-v4": (35, 210, 0, 160),
        "PongDeterministic-v4": (35, 210, 0, 160),
        "SpaceInvadersDeterministic-v4": (30, 200, 0, 160),
        "RobotankDeterministic-v4": (75, 170, 10, 160),
        "FlappyBird-v0": (0, 405, 0, 288),
        "VizdoomDefendCenter-v0": (0, 400, 0, 600),
        "VizdoomDeathmatch-v0": (0, 400, 0, 600)
    }
    
    # Get crop region for the game, or use full frame if game not specified
    if game_name and game_name in crop_regions:
        top, bottom, left, right = crop_regions[game_name]
        x = x[top:bottom, left:right]
    
    # Convert to grayscale
    if len(x.shape) == 3:
        state = skimage.color.rgb2gray(x)
    else:
        state = x
    
    # Resize to standard size
    state = skimage.transform.resize(state, (84, 84), preserve_range=True)
    
    # Normalize to [0, 255]
    state = skimage.exposure.rescale_intensity(state, out_range=(0, 255))
    state = state.astype('uint8')
    
    return state

def makeState(state: List[np.ndarray]) -> np.ndarray:
    """
    Stack frames to create a state representation.
    
    Args:
        state: List of 4 consecutive frames
        
    Returns:
        Stacked frames as a numpy array
    """
    return np.stack(state, axis=0)

def preprocess_frame(frame: np.ndarray, game_name: str = None) -> np.ndarray:
    """
    Additional preprocessing steps for frames.
    
    Args:
        frame: Raw frame from the environment
        game_name: Name of the game for game-specific preprocessing
        
    Returns:
        Preprocessed frame
    """
    # Apply game-specific preprocessing
    if game_name == "SpaceInvadersDeterministic-v4":
        # Remove score display
        frame[0:30, :] = 0
    elif game_name == "BreakoutDeterministic-v4":
        # Remove score and lives display
        frame[0:30, :] = 0
    elif game_name in ["VizdoomDefendCenter-v0", "VizdoomDeathmatch-v0"]:
        # Apply contrast enhancement for better visibility
        frame = skimage.exposure.adjust_gamma(frame, gamma=0.8)
    
    return frame

def getFrameWithPreprocessing(x: np.ndarray, game_name: str = None) -> np.ndarray:
    """
    Complete frame processing pipeline including preprocessing.
    
    Args:
        x: Raw frame from the environment
        game_name: Name of the game
        
    Returns:
        Fully processed frame
    """
    # Apply preprocessing
    x = preprocess_frame(x, game_name)
    
    # Get processed frame
    return getFrame(x, game_name)