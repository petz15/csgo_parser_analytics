# Quick Start Guide - CS:GO Economic RL

## Installation

```bash
pip install torch tqdm numpy matplotlib
```

## Quick Test

```bash
# Test the strategies
python ml_model.py
```

## Training

```bash
# Quick training (1000 matches)
python trainer.py --n-matches 1000 --eval-matches 100

# Extensive training with specific strategies
python trainer.py --strategies dqn ppo adaptive --n-matches 5000 --eval-matches 200

# Train all available strategies
python trainer.py --strategies full_buy conservative random adaptive momentum dqn ppo reinforce --n-matches 2000
```

## Export Models for Go

```bash
# Export all trained models to Go-compatible format
python export_models.py --models-dir models --output-dir exported_models
```

## Example Usage in Python

```python
from ml_model import StrategyFactory, GameState

# Create any strategy
strategy = StrategyFactory.create('ppo')  # or 'dqn', 'adaptive', etc.

# Define game state
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

# Get investment decision
investment_ratio = strategy.select_action(state, training=False)
investment_amount = state.own_funds * investment_ratio

print(f"Strategy: {strategy.name}")
print(f"Decision: Invest {investment_ratio:.1%} (${investment_amount:,.0f})")
```

## Available Strategies

### Baseline (No Learning)
- `full_buy` - Always invest 100%
- `conservative` - Always invest 50%
- `random` - Random investment
- `adaptive` - Rule-based adaptive strategy
- `momentum` - Based on recent performance

### Reinforcement Learning
- `dqn` - Deep Q-Network (discrete actions)
- `ppo` - Proximal Policy Optimization (continuous)
- `reinforce` - Simple policy gradient (continuous)

## Training Results

After training, check:
- `models/` - Saved model checkpoints
- `training_results_*.json` - Win rates and statistics

## Key Features

✓ **Continuous Action Space** - Investment ratio 0.0 to 1.0
✓ **Multiple RL Methods** - DQN, PPO, REINFORCE
✓ **Baseline Strategies** - Rule-based comparisons
✓ **Self-Play Training** - Strategies learn against each other
✓ **Go Export** - Easy integration with Go ABM
✓ **No CSF Knowledge** - Agents learn from outcomes only

## Next Steps

1. **Train models**: `python trainer.py`
2. **Evaluate performance**: Check training_results_*.json
3. **Export best model**: `python export_models.py`
4. **Integrate with Go ABM**: Use exported model files

For detailed documentation, see [RL_README.md](RL_README.md)
