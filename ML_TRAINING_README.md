# Machine Learning Training Guide

Comprehensive guide for training reinforcement learning agents on CS:GO economic decision-making using Agent-Based Model (ABM) simulation data.

## Overview

This project uses **imitation learning** to train RL agents from historical ABM simulation results. Agents learn optimal equipment spending strategies by observing game outcomes from simulated matches between different economic strategies.

### Training Architecture

```
ABM Simulations → CSV Trajectories → Experience Replay → ML Training → Trained Models
```

**Key Concept**: Agents learn from complete game trajectories (not single rounds) to understand long-term consequences of economic decisions.

## Strategy Types

### Reinforcement Learning Strategies
- **DQN** (Deep Q-Network): Value-based learning with experience replay and target networks #not recommended for ABM, too slow
- **PPO** (Proximal Policy Optimization): Policy gradient method with clipped objective
- **REINFORCE**: Basic policy gradient with baseline

### Supervised Learning Strategies
- **XGBoost**: Gradient boosted decision trees
- **Tree**: Single decision tree
- **Logistic Regression**: Linear logistic model
- **SGD**: Stochastic gradient descent classifier

## Training Data Format

### Input: ABM Simulation CSVs

CSV files contain round-by-round game state and outcomes with columns:
```
t1_funds, t1_score, t1_survivors, t1_consecutive_losses, t1_is_ct_side, 
t1_investment, t1_starting_equipment, ...
t2_funds, t2_score, t2_survivors, t2_consecutive_losses, t2_is_ct_side,
t2_investment, t2_starting_equipment, ...
round_number, last_round_reason, last_bomb_planted, t1_won_round, match_winner
```

### Game State Features

**Observable Features** (legal information):
- `own_funds`: Current team funds
- `own_score`, `opponent_score`: Round scores
- `own_survivors`, `opponent_survivors`: Living players after previous round
- `consecutive_losses`: Loss bonus accumulation
- `is_ct_side`: Team side (CT=true, T=false)
- `round_number`: Current round (1-30+)
- `half_length`: Rounds per half (usually 15)
- `last_round_reason`: Previous round outcome (bomb exploded, defused, elimination, etc.)
- `last_bomb_planted`: Whether bomb was planted last round
- `own_starting_equipment`: Equipment value at round start
- `score_diff`: Derived feature (own_score - opponent_score)

**Forbidden Features** (opponent private info, only for experimental use):
- `opponent_funds`: Opponent's money (not visible in real game)
- `opponent_starting_equipment`: Opponent's equipment value

### Action Space

**Continuous**: Investment amount as fraction of available funds [0.0, 1.0]
- 0.0 = Save (eco round)
- 0.5 = Half-buy
- 1.0 = Full-buy

### Reward Signal

**Binary match outcome**: +1 for win, -1 for loss (same for all rounds in a match)

## Training Process

### Step 1: Generate ABM Simulation Data

Run ABM simulations to create training data:
```bash
# Run ABM simulation notebook (e.g., abm_v9.ipynb)
# Outputs to matchup_XXX_strategy1_vs_strategy2/ folder
```

Expected output structure:
```
matchup_021_anti_allin_v3_vs_expected_value/
├── all_games_minimal.csv          # Main simulation data
├── simulation_summary.json        # Metadata (total games, win rates, etc.)
└── analysis_report_*.html         # Optional analysis report
```

### Step 2: Train RL Model

Basic training command:
```bash
python train_from_abm_data.py \
    --results-folder matchup_021_anti_allin_v3_vs_expected_value \
    --strategy dqn \
    --episodes 500 \
    --batch-size 128 \
    --lr 0.001 \
    --output-dir models_from_abm
```

### Step 3: Monitor Training

Training output displays:
```
📊 Loaded ABM Summary:
  Total simulations: 1000
  Total rounds: 24583
  Team1 win rate: 52.30%
  Team2 win rate: 47.70%
  Average rounds per game: 24.58

Training Episode 50/500 | Win Rate: 0.48 | Avg Reward: -0.04 | Loss: 0.234
```

### Step 4: Save and Export Models

Models are automatically saved to `models_from_abm/`:
```
models_from_abm/
├── dqn_abm_trained.pt           # PyTorch model
└── training_info.json            # Hyperparameters and metrics
```

Export to Go-compatible format:
```bash
python export_models.py --model-path models_from_abm/dqn_abm_trained.pt --output-dir ml_go_models/dqn_abm_trained
```

Outputs:
```
ml_go_models/dqn_abm_trained/
├── dqn_model.json               # Weights and biases
└── metadata.json                 # Architecture and normalization
```

## Training Options

### Common Arguments

```bash
python train_from_abm_data.py [OPTIONS]

Required:
  --results-folder PATH       ABM simulation results folder
  --strategy NAME             Strategy type: dqn, ppo, reinforce, tree, xgboost, logistic, sgd

Training Parameters:
  --episodes INT              Number of training iterations (default: 500)
  --batch-size INT            Batch size for RL training (default: 128)
  --lr FLOAT                  Learning rate (default: 0.001)
  --gamma FLOAT               Discount factor (default: 0.99)
  
Output:
  --output-dir PATH           Directory for saved models (default: models_from_abm)
  --save-interval INT         Save checkpoint every N episodes (default: 50)

Advanced:
  --use-forbidden-state       Include opponent's private information (EXPERIMENTAL)
  --replay-buffer-size INT    Experience replay buffer size (default: 10000)
  --target-update INT         Update target network every N episodes (DQN only, default: 10)
```

### Strategy-Specific Parameters

**DQN**:
```bash
--replay-buffer-size 50000    # Larger buffer for more diverse experiences
--target-update 20            # Slower target network updates for stability
--epsilon-decay 0.995         # ε-greedy exploration decay
```

**PPO**:
```bash
--ppo-epochs 10               # Optimization epochs per batch
--clip-epsilon 0.2            # PPO clipping parameter
--entropy-coef 0.01           # Entropy bonus for exploration
```

**XGBoost**:
```bash
--n-estimators 200            # Number of boosting rounds
--max-depth 6                 # Tree depth
--learning-rate 0.1           # Boosting learning rate
```

## Training Best Practices

### 1. Data Quality
- Use at least 1,000 ABM simulations for training
- Ensure balanced win rates (40-60%) between strategies
- Check for sufficient trajectory diversity

### 2. Hyperparameter Tuning
- **Learning rate**: Start with 0.001, decrease if training is unstable
- **Batch size**: Larger (256-512) for stable gradients, smaller (64-128) for faster iteration
- **Episodes**: RL strategies need 500-1000 episodes; supervised methods converge faster (100-200)

### 3. Monitoring Convergence
- Watch for plateau in win rate (indicates convergence or overfitting)
- Loss should decrease steadily; spikes indicate instability
- Validate on held-out ABM simulations

### 4. Avoiding Pitfalls
- **Don't use forbidden state** in production models (breaks game rules)
- **Normalize features** consistently (ml_model.py handles this automatically)
- **Test generalization**: Validate against unseen opponent strategies

## Training Workflow Example

Complete workflow for training DQN agent:

```bash
# 1. Generate ABM data (run abm_v9.ipynb or similar)
#    Outputs to: matchup_030_my_strategy_vs_baseline/

# 2. Train DQN model
python train_from_abm_data.py \
    --results-folder matchup_030_my_strategy_vs_baseline \
    --strategy dqn \
    --episodes 1000 \
    --batch-size 256 \
    --lr 0.0005 \
    --gamma 0.99 \
    --replay-buffer-size 50000 \
    --target-update 10 \
    --output-dir models_from_abm \
    --save-interval 100

# 3. Export to Go format
python export_models.py \
    --model-path models_from_abm/dqn_abm_trained.pt \
    --output-dir ml_go_models/dqn_final

# 4. Test in ABM simulation (use exported model in abm_v9.ipynb)

# 5. Iterate: Retrain on new data, tune hyperparameters, compare strategies
```

## Understanding the Code

### Key Classes

**ml_model.py**:
- `GameState`: Observable game state representation
- `ForbiddenGameState`: Extended state with opponent info (experimental)
- `Experience`: Single trajectory step (state, action, reward, next_state)
- `BaseStrategy`: Abstract strategy interface
- `DQNStrategy`, `PPOStrategy`, etc.: Specific RL implementations
- `StrategyFactory`: Creates strategy instances by name

**train_from_abm_data.py**:
- `ABMDataLoader`: Loads and parses ABM CSV files
- `ImitationTrainer`: Main training loop for RL agents
- `extract_trajectories()`: Converts CSV rows to Experience objects
- `train_*_strategy()`: Strategy-specific training methods

**export_models.py**:
- `export_dqn_to_json()`: Export DQN weights and architecture
- `export_xgboost_to_json()`: Export tree-based models
- `export_logistic_to_json()`: Export linear models

### Training Loop (Simplified)

```python
# Load ABM data
loader = ABMDataLoader(results_folder)
all_experiences = loader.load_all_trajectories()

# Create strategy
strategy = StrategyFactory.create(strategy_name)

# Training loop
for episode in range(num_episodes):
    # Sample batch of experiences
    batch = random.sample(all_experiences, batch_size)
    
    # Compute loss and update
    loss = strategy.train_step(batch)
    
    # Evaluate performance
    if episode % eval_interval == 0:
        win_rate = evaluate_on_abm_data(strategy, validation_set)
        print(f"Episode {episode}: Win Rate = {win_rate:.2%}")

# Save final model
torch.save(strategy.state_dict(), output_path)
```

## Troubleshooting

### Common Issues

**Problem**: Training loss not decreasing
- **Solution**: Lower learning rate (try 0.0001), increase batch size, check data quality

**Problem**: Win rate stuck at ~50%
- **Solution**: Agent may have converged; try more episodes, tune exploration parameters, or strategy isn't learning useful patterns

**Problem**: Out of memory errors
- **Solution**: Reduce batch size, decrease replay buffer size, use gradient accumulation

**Problem**: "No trajectories loaded" error
- **Solution**: Check CSV file format matches expected columns, verify results folder path

**Problem**: Model performs well in training but poorly in validation
- **Solution**: Overfitting; reduce model capacity, add regularization, increase training data diversity

### Debugging Tips

1. **Inspect trajectories**: Print first few experiences to verify state/action/reward parsing
2. **Check normalization**: Ensure features are in [0, 1] range (see `GameState.normalize()`)
3. **Validate rewards**: All experiences from winning match should have reward=+1.0
4. **Monitor gradients**: Use `torch.nn.utils.clip_grad_norm_()` if gradients explode
5. **Visualize Q-values**: Plot Q(s,a) for different states to understand learned policy


