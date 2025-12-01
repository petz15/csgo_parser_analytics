# Usage Examples - CS:GO Economic ABM RL Training

This document provides practical examples for training RL models from ABM data and deploying them.

## Table of Contents
- [Quick Start](#quick-start)
- [Training from ABM Data](#training-from-abm-data)
- [Export to Go](#export-to-go)
- [ABM Integration](#abm-integration)
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

## Export to Go

### Example 5: Export DQN Model
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

### Example 6: Export PPO Model
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

### Example 7: Export All Models
```powershell
# Export all available models
python export_models.py `
  --models-dir models_from_abm `
  --output-dir exported_go_models `
  --strategies dqn ppo reinforce
```

---

## ABM Integration

**See `ABM_INTEGRATION.md` for complete guide on using trained models in your Go ABM.**

### Example 8: Test Model Decisions

```python
# test_model_decisions.py
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
python test_model_decisions.py
```

---

## Advanced Workflows

### Example 9: Complete Training and Export Pipeline
```powershell
# 1. Train from ABM data
python train_from_abm_data.py `
  --results-dir results_20251118_002441 `
  --strategies dqn `
  --use-winning-only `
  --epochs 20 `
  --save-dir production_models

# 2. Export for Go
python export_models.py `
  --models-dir production_models `
  --output-dir go_models `
  --strategies dqn

# 3. Copy to your Go ABM project
Copy-Item go_models\dqn_model.json ..\your_go_abm\models\
Copy-Item go_models\dqn_inference.go ..\your_go_abm\
```

### Example 10: Hyperparameter Sweep
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

### Example 11: Train Multiple Strategies and Compare
```powershell
# Train DQN, PPO, and REINFORCE
python train_from_abm_data.py `
  --results-dir results_20251118_002441 `
  --strategies dqn ppo reinforce `
  --use-winning-only `
  --epochs 20 `
  --save-dir models_comparison

# View training info for each
Get-ChildItem models_comparison\*.pt | ForEach-Object {
    Write-Host "`nModel: $($_.Name)"
}
```

### Example 12: Data Augmentation
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
```

### Example 13: Production Deployment Checklist
```powershell
# 1. Train production model
python train_from_abm_data.py `
  --results-dir results_20251118_002441 `
  --strategies dqn `
  --use-winning-only `
  --epochs 50 `
  --batch-size 256 `
  --save-dir production

# 2. Export for Go
python export_models.py `
  --models-dir production `
  --output-dir production_export `
  --strategies dqn

# 3. Test in Go ABM
# (See ABM_INTEGRATION.md for Go code)

# 4. Backup models
Copy-Item -Recurse production "production_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

# 5. Document version
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
python train_from_abm_data.py --batch-size 256  # Faster if you have GPU

# Limit simulations during development
python train_from_abm_data.py --n-simulations 100  # Quick iterations

# Use winning-only for cleaner signal
python train_from_abm_data.py --use-winning-only  # Better convergence
```

### Analyzing Training Results
```powershell
# View training info
Get-Content models_from_abm/training_info.json | ConvertFrom-Json | Format-List

# Check all model files
Get-ChildItem models_from_abm/*.pt | ForEach-Object {
    Write-Host "Model: $($_.Name) - Size: $($_.Length) bytes"
}
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

1. **Start Simple**: Begin with Example 3 (quick prototype)
2. **Iterate**: Use Example 10 to find best hyperparameters
3. **Scale Up**: Move to Example 4 (full training)
4. **Deploy**: Follow Example 13 (production checklist)
5. **Integrate**: See `ABM_INTEGRATION.md` for complete Go integration guide

For more details, see:
- `ABM_INTEGRATION.md` - Complete guide for using models in Go ABM
- `RL_README.md` - RL algorithm documentation
- `QUICKSTART.md` - Quick reference
- `ml_model.py` - Source code with comments
