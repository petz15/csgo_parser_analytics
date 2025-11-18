# Usage Examples - CS:GO Economic ABM RL Training

This document provides practical examples for training and evaluating RL models.

## Table of Contents
- [Quick Start](#quick-start)
- [Training from ABM Data](#training-from-abm-data)
- [Self-Play Training](#self-play-training)
- [Model Evaluation](#model-evaluation)
- [Export to Go](#export-to-go)
- [Advanced Workflows](#advanced-workflows)

---

## Quick Start

### 1. Check your ABM data
```powershell
# List available simulation results
Get-ChildItem -Path results_* -Directory

# Check a specific results folder
Get-ChildItem -Path results_20251118_002441 | Select-Object Name
```

### 2. Train a simple model
```powershell
# Train DQN on 100 simulations (quick test)
python train_from_abm_data.py --results-dir results_20251118_002441 --n-simulations 100 --strategies dqn --epochs 5

# Check the trained model
Get-ChildItem -Path models_from_abm
```

### 3. Test the trained model
```powershell
# Load and test the model
python -c "from ml_model import StrategyFactory; import torch; strategy = StrategyFactory.create('dqn'); strategy.load('models_from_abm/dqn_abm_trained.pt'); print('Model loaded successfully!')"
```

---

## Training from ABM Data

### Example 1: Train on Winning Behaviors Only
```powershell
# Learn from successful strategies only
python train_from_abm_data.py `
  --results-dir results_20251118_002441 `
  --strategies dqn ppo `
  --use-winning-only `
  --epochs 20 `
  --batch-size 64 `
  --save-dir models_winning_only
```

**Why?** Training on winning behaviors helps the model learn what works, avoiding confusion from losing strategies.

**Output:**
- `models_winning_only/dqn_abm_trained.pt`
- `models_winning_only/ppo_abm_trained.pt`
- `models_winning_only/training_info.json`

### Example 2: Train on All Data (Balanced Learning)
```powershell
# Learn from both winning and losing behaviors
python train_from_abm_data.py `
  --results-dir results_20251118_002441 `
  --strategies dqn ppo reinforce `
  --epochs 15 `
  --batch-size 128 `
  --save-dir models_balanced
```

**Why?** Learning from both outcomes can help the model understand what NOT to do.

### Example 3: Quick Prototype (Fast Iteration)
```powershell
# Use subset of data for rapid experimentation
python train_from_abm_data.py `
  --results-dir results_20251118_002441 `
  --n-simulations 50 `
  --strategies dqn `
  --epochs 3 `
  --batch-size 32 `
  --save-dir models_prototype
```

**Why?** Perfect for testing changes without waiting hours.

### Example 4: Full Training Run (Production)
```powershell
# Use all available data with extensive training
python train_from_abm_data.py `
  --results-dir results_20251118_002441 `
  --strategies dqn ppo `
  --use-winning-only `
  --epochs 50 `
  --batch-size 256 `
  --save-dir models_production
```

**Why?** Maximize model performance with comprehensive training.

---

## Self-Play Training

### Example 5: Train Against Baselines
```powershell
# Train DQN and PPO against rule-based strategies
python trainer.py `
  --n-matches 1000 `
  --eval-matches 100 `
  --strategies dqn ppo adaptive momentum `
  --save-dir models_selfplay
```

**Output:**
- Match results and win rates
- Trained models in `models_selfplay/`
- `training_results_*.json` with performance metrics

### Example 6: Focused Self-Play (Two Strategies)
```powershell
# Train DQN vs PPO head-to-head
python trainer.py `
  --n-matches 2000 `
  --eval-matches 200 `
  --strategies dqn ppo `
  --save-dir models_dqn_vs_ppo
```

**Why?** Focused competition can lead to more specialized strategies.

### Example 7: Tournament Style (All Strategies)
```powershell
# Round-robin tournament with all strategies
python trainer.py `
  --n-matches 500 `
  --eval-matches 100 `
  --strategies fullbuy conservative random adaptive momentum dqn ppo reinforce `
  --save-dir models_tournament
```

**Why?** Find the most robust strategy across diverse opponents.

---

## Model Evaluation

### Example 8: Compare Trained Models
```powershell
# Evaluate models trained from ABM data vs self-play
python trainer.py `
  --n-matches 0 `
  --eval-matches 500 `
  --strategies dqn ppo adaptive `
  --save-dir evaluation_results

# Load pre-trained models first (modify trainer.py to load from specific paths)
```

### Example 9: Analyze Training Progress
```powershell
# View training info
Get-Content models_from_abm/training_info.json | ConvertFrom-Json | Format-List

# Check all model files
Get-ChildItem models_from_abm/*.pt | ForEach-Object {
    Write-Host "Model: $($_.Name) - Size: $($_.Length) bytes"
}
```

### Example 10: Test Model Inference
```python
# test_inference.py
from ml_model import StrategyFactory, GameState

# Load trained model
strategy = StrategyFactory.create('dqn')
strategy.load('models_from_abm/dqn_abm_trained.pt')

# Test on various game states
test_states = [
    # Early game, low funds
    GameState(5000, 0, 0, 5, 5, 0, True, 1, 15, 0, False),
    
    # Mid game, winning
    GameState(20000, 8, 3, 4, 2, 0, True, 12, 15, 1, False),
    
    # Late game, losing, low funds
    GameState(3000, 7, 10, 2, 5, 3, False, 18, 15, 0, False),
]

for i, state in enumerate(test_states):
    action = strategy.select_action(state, training=False)
    investment = int(action * state.own_funds)
    print(f"State {i+1}: Invest {action*100:.1f}% (${investment:,})")
```

```powershell
# Run the test
python test_inference.py
```

---

## Export to Go

### Example 11: Export DQN Model
```powershell
# Export trained DQN model
python export_models.py `
  --models-dir models_from_abm `
  --output-dir exported_go_models `
  --strategies dqn
```

**Output:**
- `exported_go_models/dqn_model.json` - Weights and metadata
- `exported_go_models/dqn_inference.go` - Go inference code

### Example 12: Export PPO Model
```powershell
# Export trained PPO model
python export_models.py `
  --models-dir models_from_abm `
  --output-dir exported_go_models `
  --strategies ppo
```

**Output:**
- `exported_go_models/ppo_model.json`
- `exported_go_models/ppo_inference.go`

### Example 13: Export All Models
```powershell
# Export all available models
python export_models.py `
  --models-dir models_from_abm `
  --output-dir exported_go_models `
  --strategies dqn ppo reinforce
```

### Example 14: Integrate with Go ABM
```go
// In your Go ABM code
package main

import (
    "encoding/json"
    "os"
)

// Load the exported model
func loadModel(path string) (*DQNModel, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, err
    }
    
    var model DQNModel
    if err := json.Unmarshal(data, &model); err != nil {
        return nil, err
    }
    
    return &model, nil
}

func main() {
    // Load model
    model, err := loadModel("exported_go_models/dqn_model.json")
    if err != nil {
        panic(err)
    }
    
    // Create game state
    state := GameState{
        OwnFunds: 10000,
        OwnScore: 5,
        OpponentScore: 4,
        // ... other fields
    }
    
    // Get investment decision
    action := model.Forward(state.ToArray())
    investment := int(action * float64(state.OwnFunds))
    
    println("AI Decision: Invest $", investment)
}
```

---

## Advanced Workflows

### Example 15: Continuous Training Pipeline
```powershell
# 1. Initial training from ABM data
python train_from_abm_data.py `
  --results-dir results_20251118_002441 `
  --strategies dqn ppo `
  --use-winning-only `
  --epochs 20 `
  --save-dir models_stage1

# 2. Refine with self-play
python trainer.py `
  --n-matches 2000 `
  --strategies dqn ppo `
  --save-dir models_stage2
  # (Manually load stage1 models first)

# 3. Final evaluation
python trainer.py `
  --n-matches 0 `
  --eval-matches 1000 `
  --strategies dqn ppo adaptive `
  --save-dir models_final_eval

# 4. Export best model
python export_models.py `
  --models-dir models_stage2 `
  --output-dir production_models `
  --strategies dqn
```

### Example 16: Hyperparameter Sweep
```powershell
# Test different batch sizes
foreach ($batch in @(32, 64, 128, 256)) {
    python train_from_abm_data.py `
      --results-dir results_20251118_002441 `
      --strategies dqn `
      --batch-size $batch `
      --epochs 10 `
      --save-dir "models_batch_$batch"
}

# Compare results
Get-ChildItem models_batch_* -Directory | ForEach-Object {
    Write-Host "`n$($_.Name):"
    Get-Content "$($_.FullName)/training_info.json" | ConvertFrom-Json | Format-List
}
```

### Example 17: Multi-Stage Training
```powershell
# Stage 1: Learn from data (supervised)
python train_from_abm_data.py `
  --results-dir results_20251118_002441 `
  --strategies dqn `
  --use-winning-only `
  --epochs 30 `
  --save-dir models_supervised

# Stage 2: Improve via self-play (reinforcement)
python trainer.py `
  --n-matches 5000 `
  --strategies dqn adaptive `
  --save-dir models_refined
  # Load models_supervised/dqn_abm_trained.pt first

# Stage 3: Final polish against diverse opponents
python trainer.py `
  --n-matches 3000 `
  --strategies dqn fullbuy conservative random adaptive momentum `
  --save-dir models_polished
```

### Example 18: A/B Testing Different Strategies
```python
# compare_strategies.py
from ml_model import StrategyFactory
import json

strategies = {
    'abm_trained': 'models_from_abm/dqn_abm_trained.pt',
    'selfplay': 'models_selfplay/dqn_final.pt',
    'baseline': None  # Fresh initialization
}

results = {}
for name, path in strategies.items():
    strategy = StrategyFactory.create('dqn')
    if path:
        strategy.load(path)
    
    # Test on same state
    test_state = GameState(15000, 7, 7, 3, 3, 1, True, 15, 15, 0, False)
    action = strategy.select_action(test_state, training=False)
    
    results[name] = {
        'action': float(action),
        'investment': int(action * test_state.own_funds)
    }

print(json.dumps(results, indent=2))
```

```powershell
python compare_strategies.py
```

### Example 19: Data Augmentation
```powershell
# Train on multiple result folders (if you have multiple ABM runs)
python train_from_abm_data.py `
  --results-dir results_20251118_002441 `
  --strategies dqn `
  --epochs 20 `
  --save-dir models_dataset1

python train_from_abm_data.py `
  --results-dir results_20251119_120000 `
  --strategies dqn `
  --epochs 20 `
  --save-dir models_dataset2

# Combine insights from both
```

### Example 20: Production Deployment Checklist
```powershell
# 1. Train production model
python train_from_abm_data.py `
  --results-dir results_20251118_002441 `
  --strategies dqn `
  --use-winning-only `
  --epochs 50 `
  --batch-size 256 `
  --save-dir production

# 2. Validate performance
python trainer.py `
  --n-matches 0 `
  --eval-matches 2000 `
  --strategies dqn adaptive `
  --save-dir production_eval

# 3. Export for Go
python export_models.py `
  --models-dir production `
  --output-dir production_export `
  --strategies dqn

# 4. Test Go integration
go run your_abm.go --model production_export/dqn_model.json

# 5. Backup models
Copy-Item -Recurse production "production_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

# 6. Document version
@{
    version = "1.0.0"
    trained_date = (Get-Date).ToString()
    data_source = "results_20251118_002441"
    strategy = "dqn"
    epochs = 50
} | ConvertTo-Json | Out-File production/version.json
```

---

## Tips and Best Practices

### Performance Tips
```powershell
# Use batch processing for large datasets
python train_from_abm_data.py --batch-size 256  # Faster on GPU

# Limit simulations during development
python train_from_abm_data.py --n-simulations 100  # Quick iterations

# Use winning-only for cleaner signal
python train_from_abm_data.py --use-winning-only  # Better convergence
```

### Debugging
```powershell
# Check data loading
python -c "from train_from_abm_data import ABMDataLoader; loader = ABMDataLoader('results_20251118_002441'); print(f'Found {len(loader.sim_files)} files')"

# Test single strategy
python train_from_abm_data.py --strategies dqn --n-simulations 10 --epochs 1

# Verify model saves
Test-Path models_from_abm/dqn_abm_trained.pt
```

### Common Issues
```powershell
# Issue: Out of memory
# Solution: Reduce batch size
python train_from_abm_data.py --batch-size 32

# Issue: Slow training
# Solution: Reduce epochs or simulations
python train_from_abm_data.py --epochs 5 --n-simulations 200

# Issue: Poor performance
# Solution: Train longer or use winning-only
python train_from_abm_data.py --epochs 30 --use-winning-only
```

---

## Next Steps

1. **Start Simple**: Begin with Example 2 (quick prototype)
2. **Iterate**: Use Example 16 to find best hyperparameters
3. **Scale Up**: Move to Example 4 (full training)
4. **Deploy**: Follow Example 20 (production checklist)

For more details, see:
- `RL_README.md` - Comprehensive documentation
- `QUICKSTART.md` - Quick reference
- `ml_model.py` - Source code with comments
