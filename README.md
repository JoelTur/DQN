# Deep Q-Network (DQN) for Atari Games

This project implements a Deep Q-Network (DQN) agent for playing Atari games using PyTorch. The implementation includes prioritized experience replay, frame stacking, and various optimizations for better training performance.


## Features

- Deep Q-Network implementation with prioritized experience replay
- Convolutional Neural Network architecture optimized for Atari games
- Frame stacking and preprocessing for better state representation
- Configurable hyperparameters for different games
- Weights & Biases integration for experiment tracking
- Comprehensive logging and visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/youruname/DQN.git
cd DQN
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the agent on a specific game:

```bash
python TRAIN_AGENT.py --config config_name --game game_name
```

Example:
```bash
python TRAIN_AGENT.py --config default --game BreakoutDeterministic-v4
```

### Testing

To test a trained agent:

```bash
python TEST_AGENT.py --config config_name --game game_name --model_path path/to/model.pth
```

Example:
```bash
python TEST_AGENT.py --config default --game BreakoutDeterministic-v4 --model_path models/breakout_model.pth
```

## Configuration

The project uses a configuration system to manage hyperparameters and settings:

1. `config/config.py`: Contains general configuration settings
2. `config/hyperparameters.py`: Contains game-specific hyperparameters

To add a new game configuration:
1. Add game-specific hyperparameters in `config/hyperparameters.py`
2. Create a new configuration in `config/config.py`

## Model Architecture

The CNN architecture includes:
- 3 convolutional layers with batch normalization and ReLU activation
- Dropout layers for regularization
- Fully connected layers for Q-value prediction

## Training Process

1. Frame Preprocessing:
   - Grayscale conversion
   - Resizing to 84x84
   - Normalization to [0, 255]

2. Experience Collection:
   - Frame stacking (4 frames)
   - Prioritized experience replay
   - Epsilon-greedy exploration

3. Training:
   - Target network for stable learning
   - Huber loss for robust training
   - Gradient clipping for stability

## Results

The agent's performance is tracked using:
- Average reward per episode
- Training loss
- Epsilon value
- Frame processing time

Results are logged to:
- Console output
- Log files
- Weights & Biases (if enabled)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DeepMind's original DQN paper
- PyTorch team for the excellent framework
- Gymnasium team for the Atari environments 
