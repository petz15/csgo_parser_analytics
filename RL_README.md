# CS:GO Economic ABM - Reinforcement Learning Models

This project implements various reinforcement learning strategies for optimal economic decision-making in a simplified CS:GO Agent-Based Model (ABM) where teams decide how much to invest in equipment each round.

## Overview

The game is an abstract version of CS:GO focusing only on the economic aspect. Teams (agents) make decisions about equipment investment without knowing the underlying Contest Success Function (CSF) or opponent's strategy - they must learn optimal behavior through experience.

### Observable Information
Each team observes:
- Own team's current money
- Own team's score
- Opponent's score
- Number of survivors (both teams) from previous round
- Round end reason (1=T Bomb, 2=T Elim, 3=CT Defuse, 4=CT Elim)
- Bomb plant status
- Consecutive loss bonus counter
- Current side (CT/T)
- Round number and half length

### Decision Space
- **Action**: Continuous investment ratio (0.0 to 1.0 of available funds)
- **Objective**: Maximize match win probability

## Implemented Strategies

### Baseline Strategies (No Learning)
1. **FullBuy**: Always invest 100% of available funds
2. **Conservative**: Always invest 50% of available funds
3. **Random**: Random investment between 0-100%
4. **AdaptiveThreshold**: Rule-based strategy considering score, funds, and consecutive losses
5. **Momentum**: Investment based on recent performance (survivor counts)

### Reinforcement Learning Strategies

#### 1. Deep Q-Network (DQN)
- **Type**: Value-based RL
- **Action Space**: Discretized (11 actions: 0%, 10%, 20%, ..., 100%)
- **Features**:
  - Experience replay buffer
  - Target network for stability
  - Epsilon-greedy exploration
  - Double Q-learning

#### 2. Proximal Policy Optimization (PPO)
- **Type**: Policy gradient
- **Action Space**: Continuous (Beta distribution)
- **Features**:
  - Actor-Critic architecture
  - Generalized Advantage Estimation (GAE)
  - Clipped surrogate objective
  - Multiple epochs per update
  - Works well with continuous actions

#### 3. REINFORCE
- **Type**: Simple policy gradient
- **Action Space**: Continuous (Beta distribution)
- **Features**:
  - Monte Carlo returns
  - Simpler than PPO but less stable
  - Good baseline for policy gradient methods

## Project Structure

```
├── ml_model.py           # RL strategy implementations
├── trainer.py            # Training script
├── export_models.py      # Export to Go-compatible format
├── distributions.json    # ABM outcome distributions
└── results_*/            # ABM simulation results
```

## Usage

### 1. Train Strategies

```bash
# Train default strategies (full_buy, adaptive, dqn, ppo)
python trainer.py --n-matches 1000 --eval-matches 100

# Train specific strategies
python trainer.py --strategies dqn ppo reinforce --n-matches 2000

# Skip training, only evaluate existing models
python trainer.py --no-train --eval-matches 200
```

### 2. Standalone Testing

```python
from ml_model import StrategyFactory, GameState

# Create a strategy
strategy = StrategyFactory.create('ppo')

# Create a game state
state = GameState(
    own_funds=10000,
    own_score=7,
    opponent_score=5,
    own_survivors=3,
    opponent_survivors=2,
    consecutive_losses=1,
    is_ct_side=True,
    round_number=12,
    half_length=15,
    last_round_reason=4,
    last_bomb_planted=False
)

# Get action
investment_ratio = strategy.select_action(state)
investment_amount = state.own_funds * investment_ratio

print(f"Invest {investment_ratio:.1%} (${investment_amount:,.0f})")
```

### 3. Export Models for Go

```bash
# Export all trained models
python export_models.py --models-dir models --output-dir exported_models
```

This generates:
- `metadata.json`: Model architecture and configuration
- `*_weights.json`: Network weights in JSON format
- `inference.go`: Go code template for inference

## Training Process

### Environment Simulation
The training environment (`TrainingEnvironment` in `trainer.py`) simulates matches between strategies:

1. **Initialization**: Both teams start with $4,000 (pistol round)
2. **Each Round**:
   - Create `GameState` for both teams
   - Strategies select investment ratios
   - Calculate equipment values
   - Simulate round outcome using CSF
   - Update economy (win rewards, loss bonuses)
   - Store experiences for learning
3. **Match End**: Determine winner, update strategies with match outcome

### Reward Structure
- **Round Reward**: +1 for win, -1 for loss
- **Match Reward**: +10 for match win, -10 for match loss
- All experiences in a match receive the final match outcome

### Training Loop
1. Randomly pair strategies (self-play or vs others)
2. Simulate match
3. Collect experiences
4. Update strategies with experiences
5. Periodic evaluation
6. Save models

## Go Implementation

The exported models can be implemented in Go using the generated templates. Key components:

### 1. Model Loading
```go
model, err := LoadModel("metadata.json", "q_network_weights.json")
```

### 2. Inference
```go
state := GameState{
    OwnFunds: 10000,
    OwnScore: 7,
    // ... other fields
}

action := model.SelectAction(state)  // Returns investment ratio 0.0-1.0
```

### 3. Neural Network Operations
The Go templates include:
- Linear layer forward pass
- ReLU activation
- Layer normalization
- Beta distribution sampling (for PPO)

## Model Architecture

### DQN
```
Input (11 features)
  ↓
Linear(11 → 128) + ReLU + LayerNorm
  ↓
Linear(128 → 128) + ReLU + LayerNorm
  ↓
Linear(128 → 64) + ReLU + LayerNorm
  ↓
Linear(64 → 11)  [Q-values for 11 actions]
```

### PPO
```
Input (11 features)
  ↓
Shared: Linear(11 → 128) + ReLU + LayerNorm
  ↓
Shared: Linear(128 → 128) + ReLU + LayerNorm
  ↓
Shared: Linear(128 → 64) + ReLU + LayerNorm
  ↓         ↓
Alpha Head  Beta Head  [Beta distribution parameters]
```

## State Features (Normalized)

1. **own_funds** / 50000 → [0, 1]
2. **own_score** / 16 → [0, 1]
3. **opponent_score** / 16 → [0, 1]
4. **own_survivors** / 5 → [0, 1]
5. **opponent_survivors** / 5 → [0, 1]
6. **consecutive_losses** / 5 → [0, 1] (capped at 5)
7. **is_ct_side** → {0, 1}
8. **round_number** / 30 → [0, 1]
9. **half_length** / 15 → [0, 1]
10. **last_round_reason** / 4 → [0, 0.25, 0.5, 0.75, 1]
11. **last_bomb_planted** → {0, 1}

## Hyperparameters

### DQN
- Learning rate: 0.0003
- Discount factor (γ): 0.99
- Epsilon decay: 0.9995
- Batch size: 64
- Replay buffer: 50,000
- Target update frequency: 100 steps

### PPO
- Learning rate: 0.0003
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- Epsilon clip: 0.2
- Epochs per update: 10
- Batch size: 64

### REINFORCE
- Learning rate: 0.001
- Discount factor (γ): 0.99

## Performance Metrics

Strategies are evaluated on:
1. **Win Rate**: Percentage of matches won
2. **Average Loss**: Training loss (for RL methods)
3. **Head-to-Head**: Round-robin evaluation against all strategies

## Future Improvements

1. **Enhanced State Representation**:
   - Track equipment history
   - Model opponent strategy
   - Consider map-specific factors

2. **Advanced RL Methods**:
   - Soft Actor-Critic (SAC)
   - TD3 (Twin Delayed DDPG)
   - Multi-agent reinforcement learning

3. **Integration with ABM**:
   - Use actual CSF distributions from `distributions.json`
   - Train on historical simulation data
   - Imitation learning from top-performing simulations

4. **Go Optimization**:
   - Compile to native Go neural network library
   - Optimize inference speed
   - Parallel strategy evaluation

## Dependencies

```bash
pip install torch numpy matplotlib tqdm
```

## License

This project is part of the CS:GO Parser Analytics research.

## Citation

If you use this code in your research, please cite:

```
@software{csgo_econ_rl_2025,
  title={CS:GO Economic ABM - Reinforcement Learning Strategies},
  author={Your Name},
  year={2025},
  url={https://github.com/petz15/csgo_parser_analytics}
}
```
