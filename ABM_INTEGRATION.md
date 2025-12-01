# Integrating Trained RL Models into Your ABM

This guide explains how to use the trained reinforcement learning models in your CS:GO economic Agent-Based Model (ABM).

## Overview

Your ABM is written in Go and simulates CS:GO economic decisions. The trained Python models need to be:
1. **Exported** to JSON format with weights
2. **Implemented** in Go for inference
3. **Integrated** into your ABM decision-making logic

---

## Step 1: Train Your Model

Train a model using your ABM simulation data:

```powershell
# Train DQN on all winning behaviors
python train_from_abm_data.py `
  --results-dir results_20251118_002441 `
  --strategies dqn `
  --use-winning-only `
  --epochs 20 `
  --save-dir trained_models
```

**Output:**
- `trained_models/dqn_abm_trained.pt` - Trained PyTorch model
- `trained_models/training_info.json` - Training metadata

---

## Step 2: Export Model to Go-Compatible Format

Export the trained model to JSON with Go inference code:

```powershell
# Export trained DQN model
python export_models.py `
  --models-dir trained_models `
  --output-dir go_models `
  --strategies dqn
```

**Output:**
- `go_models/dqn_model.json` - Model weights in JSON format
- `go_models/dqn_inference.go` - Go inference code template

---

## Step 3: Understand the Model Interface

### Input (Game State)
The model expects 11 normalized features:

```go
type GameState struct {
    OwnFunds          float64  // Current team funds (normalized: /50000)
    OwnScore          float64  // Rounds won by team (normalized: /16)
    OpponentScore     float64  // Rounds won by opponent (normalized: /16)
    OwnSurvivors      float64  // Previous round survivors (normalized: /5)
    OpponentSurvivors float64  // Opponent survivors (normalized: /5)
    ConsecutiveLosses float64  // Loss streak (normalized: /5)
    IsCtSide          float64  // 1.0 if CT, 0.0 if T
    RoundNumber       float64  // Current round (normalized: /30)
    HalfLength        float64  // Rounds per half (normalized: /15)
    LastRoundReason   float64  // Win reason code (normalized: /4)
    LastBombPlanted   float64  // 1.0 if bomb planted last round, 0.0 otherwise
}
```

### Output (Investment Decision)
The model returns a single float value:
- **Range**: [0.0, 1.0]
- **Meaning**: Fraction of available funds to invest in equipment
- **Example**: 0.75 means invest 75% of available funds

---

## Step 4: Integrate into Go ABM

### Option A: Copy Generated Code

The `export_models.py` script generates complete Go code in `dqn_inference.go`. You can:

1. Copy the generated structs and functions into your ABM
2. Load the JSON model file at startup
3. Call the inference function when making economic decisions

### Option B: Manual Integration

Here's how to integrate manually into your ABM:

```go
package main

import (
    "encoding/json"
    "io/ioutil"
    "math"
)

// Model weights structure
type DQNModel struct {
    Fc1Weight [][]float64 `json:"fc1.weight"`
    Fc1Bias   []float64   `json:"fc1.bias"`
    Fc2Weight [][]float64 `json:"fc2.weight"`
    Fc2Bias   []float64   `json:"fc2.bias"`
    Fc3Weight [][]float64 `json:"fc3.weight"`
    Fc3Bias   []float64   `json:"fc3.bias"`
    OutWeight [][]float64 `json:"out.weight"`
    OutBias   []float64   `json:"out.bias"`
}

// Game state for inference
type GameState struct {
    OwnFunds          float64
    OwnScore          float64
    OpponentScore     float64
    OwnSurvivors      float64
    OpponentSurvivors float64
    ConsecutiveLosses float64
    IsCtSide          float64
    RoundNumber       float64
    HalfLength        float64
    LastRoundReason   float64
    LastBombPlanted   float64
}

// Load model from JSON file
func LoadModel(path string) (*DQNModel, error) {
    data, err := ioutil.ReadFile(path)
    if err != nil {
        return nil, err
    }
    
    var model DQNModel
    if err := json.Unmarshal(data, &model); err != nil {
        return nil, err
    }
    
    return &model, nil
}

// Normalize game state (same as Python training)
func (s *GameState) Normalize() []float64 {
    return []float64{
        s.OwnFunds / 50000.0,
        s.OwnScore / 16.0,
        s.OpponentScore / 16.0,
        s.OwnSurvivors / 5.0,
        s.OpponentSurvivors / 5.0,
        s.ConsecutiveLosses / 5.0,
        s.IsCtSide,
        s.RoundNumber / 30.0,
        s.HalfLength / 15.0,
        s.LastRoundReason / 4.0,
        s.LastBombPlanted,
    }
}

// ReLU activation
func relu(x float64) float64 {
    if x > 0 {
        return x
    }
    return 0
}

// Matrix-vector multiplication + bias
func linear(input []float64, weight [][]float64, bias []float64) []float64 {
    output := make([]float64, len(bias))
    for i := range output {
        sum := bias[i]
        for j, val := range input {
            sum += val * weight[i][j]
        }
        output[i] = sum
    }
    return output
}

// Forward pass through DQN network
func (m *DQNModel) Forward(state []float64) float64 {
    // Layer 1
    h1 := linear(state, m.Fc1Weight, m.Fc1Bias)
    for i := range h1 {
        h1[i] = relu(h1[i])
    }
    
    // Layer 2
    h2 := linear(h1, m.Fc2Weight, m.Fc2Bias)
    for i := range h2 {
        h2[i] = relu(h2[i])
    }
    
    // Layer 3
    h3 := linear(h2, m.Fc3Weight, m.Fc3Bias)
    for i := range h3 {
        h3[i] = relu(h3[i])
    }
    
    // Output layer (Q-values for discrete actions)
    qValues := linear(h3, m.OutWeight, m.OutBias)
    
    // Find action with max Q-value
    maxIdx := 0
    maxVal := qValues[0]
    for i, val := range qValues {
        if val > maxVal {
            maxVal = val
            maxIdx = i
        }
    }
    
    // Convert discrete action to continuous [0, 1]
    // Assuming 11 discrete actions (0%, 10%, 20%, ..., 100%)
    return float64(maxIdx) / 10.0
}

// Get investment decision from AI model
func (m *DQNModel) GetInvestmentDecision(state GameState) float64 {
    normalized := state.Normalize()
    return m.Forward(normalized)
}
```

---

## Step 5: Use in Your ABM Simulation

### In Your Main ABM Code:

```go
package main

import (
    "fmt"
    "log"
)

func main() {
    // Load the trained model
    model, err := LoadModel("go_models/dqn_model.json")
    if err != nil {
        log.Fatal("Failed to load model:", err)
    }
    
    fmt.Println("✓ AI Model loaded successfully")
    
    // Run your ABM simulation
    runSimulation(model)
}

func runSimulation(model *DQNModel) {
    // Your existing ABM simulation loop
    for round := 1; round <= 30; round++ {
        // Get current game state for Team 1
        state := GameState{
            OwnFunds:          team1.Funds,
            OwnScore:          float64(team1.RoundsWon),
            OpponentScore:     float64(team2.RoundsWon),
            OwnSurvivors:      float64(team1.LastSurvivors),
            OpponentSurvivors: float64(team2.LastSurvivors),
            ConsecutiveLosses: float64(team1.ConsecLosses),
            IsCtSide:          boolToFloat(team1.IsCT),
            RoundNumber:       float64(round),
            HalfLength:        15.0,
            LastRoundReason:   float64(lastRoundReason),
            LastBombPlanted:   boolToFloat(lastBombPlanted),
        }
        
        // Get AI decision
        investmentRatio := model.GetInvestmentDecision(state)
        investmentAmount := int(investmentRatio * team1.Funds)
        
        fmt.Printf("Round %d - Team1 AI Decision: Invest %.1f%% ($%d)\n",
            round, investmentRatio*100, investmentAmount)
        
        // Apply decision in your ABM
        team1.Equipment = investmentAmount
        team1.Funds -= investmentAmount
        
        // Continue with rest of round simulation...
        simulateRound(&team1, &team2)
    }
}

func boolToFloat(b bool) float64 {
    if b {
        return 1.0
    }
    return 0.0
}
```

---

## Step 6: Compare AI vs. Baseline Strategies

You can run your ABM with different strategies:

```go
// Strategy interface
type Strategy interface {
    GetInvestmentDecision(state GameState) float64
}

// AI Strategy (trained model)
type AIStrategy struct {
    model *DQNModel
}

func (s *AIStrategy) GetInvestmentDecision(state GameState) float64 {
    return s.model.GetInvestmentDecision(state)
}

// Baseline: Always full buy
type FullBuyStrategy struct{}

func (s *FullBuyStrategy) GetInvestmentDecision(state GameState) float64 {
    return 1.0 // Always 100%
}

// Baseline: Conservative 50%
type ConservativeStrategy struct{}

func (s *ConservativeStrategy) GetInvestmentDecision(state GameState) float64 {
    return 0.5 // Always 50%
}

// Run comparison
func compareStrategies() {
    model, _ := LoadModel("go_models/dqn_model.json")
    
    strategies := map[string]Strategy{
        "AI":           &AIStrategy{model: model},
        "FullBuy":      &FullBuyStrategy{},
        "Conservative": &ConservativeStrategy{},
    }
    
    for name, strategy := range strategies {
        results := runSimulationWithStrategy(strategy, 1000)
        fmt.Printf("%s Strategy - Win Rate: %.2f%%\n", name, results.WinRate*100)
    }
}
```

---

## Step 7: Validate Model Performance

### Check Model Predictions

```go
func testModel() {
    model, _ := LoadModel("go_models/dqn_model.json")
    
    testStates := []GameState{
        // Early game, low funds
        {OwnFunds: 5000, OwnScore: 0, OpponentScore: 0, OwnSurvivors: 5, 
         OpponentSurvivors: 5, ConsecutiveLosses: 0, IsCtSide: 1, 
         RoundNumber: 1, HalfLength: 15, LastRoundReason: 0, LastBombPlanted: 0},
        
        // Mid game, winning, high funds
        {OwnFunds: 25000, OwnScore: 8, OpponentScore: 3, OwnSurvivors: 4, 
         OpponentSurvivors: 2, ConsecutiveLosses: 0, IsCtSide: 1, 
         RoundNumber: 12, HalfLength: 15, LastRoundReason: 1, LastBombPlanted: 0},
        
        // Late game, losing, low funds
        {OwnFunds: 3000, OwnScore: 7, OpponentScore: 12, OwnSurvivors: 2, 
         OpponentSurvivors: 5, ConsecutiveLosses: 3, IsCtSide: 0, 
         RoundNumber: 20, HalfLength: 15, LastRoundReason: 2, LastBombPlanted: 1},
    }
    
    for i, state := range testStates {
        decision := model.GetInvestmentDecision(state)
        investment := int(decision * state.OwnFunds)
        fmt.Printf("Scenario %d: Invest %.1f%% ($%d of $%.0f)\n",
            i+1, decision*100, investment, state.OwnFunds)
    }
}
```

---

## Complete Workflow

```powershell
# 1. Train model from ABM data
python train_from_abm_data.py --results-dir results_20251118_002441 --strategies dqn --use-winning-only --epochs 20

# 2. Export to Go
python export_models.py --models-dir models_from_abm --output-dir go_models --strategies dqn

# 3. Integrate into your Go ABM (manual step - copy code)

# 4. Build and run your Go ABM
go build -o csgo_abm.exe your_abm.go
.\csgo_abm.exe --strategy ai --model go_models/dqn_model.json

# 5. Compare with baseline strategies
.\csgo_abm.exe --strategy ai --model go_models/dqn_model.json --runs 1000
.\csgo_abm.exe --strategy fullbuy --runs 1000
.\csgo_abm.exe --strategy conservative --runs 1000
```

---

## Expected Model Behavior

Based on training from your ABM data (57% win rate for Team1), the model should learn:

1. **Early rounds**: Conservative spending (eco rounds)
2. **After wins**: Aggressive investment (capitalize on advantage)
3. **After losses**: Save or force-buy based on loss streak
4. **High funds**: Full buys when affordable
5. **Low funds + behind**: Eco to save for better round
6. **Match point**: Aggressive spending (all-in)

---

## Troubleshooting

### Model predicts unrealistic values
- Check normalization matches training (divide by same constants)
- Verify JSON loaded correctly
- Test with known states from training data

### Poor performance in ABM
- Train longer (more epochs)
- Use `--use-winning-only` flag
- Collect more ABM simulation data
- Try different strategies (ppo instead of dqn)

### Go compilation errors
- Check generated code for syntax
- Ensure JSON structure matches model export
- Verify float64 conversions

---

## Next Steps

1. **Start simple**: Integrate AI for one team, baseline for other
2. **Validate**: Run 1000+ simulations and compare win rates
3. **Iterate**: Retrain with new ABM data incorporating AI decisions
4. **Deploy**: Use best model in production ABM

For more details:
- See `export_models.py` for export options
- See `ml_model.py` for model architectures
- See generated `go_models/dqn_inference.go` for full Go implementation
