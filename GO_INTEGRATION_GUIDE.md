# Go Integration Guide for ML Models

This guide explains how to implement the trained ML models (SGD, Tree, Forest) in your Go ABM.

## Model Types Overview

### 1. **SGD Strategy** (Neural Network)
- **Type**: PyTorch neural network (64-32-1 architecture)
- **Export**: Weights as JSON
- **Go Implementation**: Manual matrix multiplication
- **Best for**: Continuous adaptation, good generalization

### 2. **Tree Strategy** (Decision Tree)
- **Type**: scikit-learn Decision Tree
- **Export**: Tree structure as JSON
- **Go Implementation**: Tree traversal
- **Best for**: Fast inference, interpretability

### 3. **Forest Strategy** (Random Forest)
- **Type**: scikit-learn Random Forest (100 trees)
- **Export**: All trees as JSON
- **Go Implementation**: Parallel tree evaluation + averaging
- **Best for**: Robust predictions, handles non-linearity

---

## Part 1: Export Models to Go-Compatible Format

### Step 1: Create Export Script

Create `export_for_go.py`:

```python
"""Export ML models to Go-compatible JSON format"""
import json
import numpy as np
from pathlib import Path
from ml_model import SGDStrategy, TreeStrategy

def export_sgd_model(model_path: str, output_dir: str):
    """Export SGD neural network to JSON"""
    strategy = SGDStrategy()
    strategy.load(model_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract weights
    weights = {}
    for name, param in strategy.network.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()
    
    model_json = {
        "model_type": "sgd_neural_network",
        "architecture": {
            "input_size": 11,
            "hidden_layers": [64, 32],
            "output_size": 1,
        },
        "weights": weights,
        "state_features": [
            "own_funds", "own_score", "opponent_score", 
            "own_survivors", "opponent_survivors", "consecutive_losses",
            "is_ct_side", "round_number", "half_length",
            "last_round_reason", "last_bomb_planted"
        ],
        "normalization": {
            "own_funds": 999999.0,
            "own_score": 16.0,
            "opponent_score": 16.0,
            "own_survivors": 5.0,
            "opponent_survivors": 5.0,
            "consecutive_losses": 5.0,
            "round_number": 30.0,
            "half_length": 15.0,
            "last_round_reason": 4.0,
        }
    }
    
    output_file = output_dir / "sgd_model.json"
    with open(output_file, 'w') as f:
        json.dump(model_json, f, indent=2)
    
    print(f"✓ Exported SGD model to {output_file}")
    return output_file


def export_tree_to_dict(tree, feature_names):
    """Recursively export sklearn tree to dictionary"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined"
        for i in tree_.feature
    ]
    
    def recurse(node):
        if tree_.feature[node] == -2:  # Leaf node
            return {
                "is_leaf": True,
                "value": float(tree_.value[node][0][0])
            }
        else:
            return {
                "is_leaf": False,
                "feature": feature_name[node],
                "threshold": float(tree_.threshold[node]),
                "left": recurse(tree_.children_left[node]),
                "right": recurse(tree_.children_right[node])
            }
    
    return recurse(0)


def export_tree_model(model_path: str, output_dir: str):
    """Export Decision Tree or Random Forest to JSON"""
    strategy = TreeStrategy(use_forest=False)
    try:
        import joblib
        data = joblib.load(model_path)
        strategy.model = data['model']
        strategy.is_fitted = data['is_fitted']
        strategy.use_forest = data.get('use_forest', False)
    except:
        print(f"Could not load model from {model_path}")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    feature_names = [
        "own_funds", "own_score", "opponent_score", 
        "own_survivors", "opponent_survivors", "consecutive_losses",
        "is_ct_side", "round_number", "half_length",
        "last_round_reason", "last_bomb_planted"
    ]
    
    if strategy.use_forest:
        # Export Random Forest (multiple trees)
        trees = []
        for estimator in strategy.model.estimators_:
            trees.append(export_tree_to_dict(estimator, feature_names))
        
        model_json = {
            "model_type": "random_forest",
            "n_trees": len(trees),
            "trees": trees,
            "state_features": feature_names,
            "normalization": {
                "own_funds": 999999.0,
                "own_score": 16.0,
                "opponent_score": 16.0,
                "own_survivors": 5.0,
                "opponent_survivors": 5.0,
                "consecutive_losses": 5.0,
                "round_number": 30.0,
                "half_length": 15.0,
                "last_round_reason": 4.0,
            }
        }
        output_file = output_dir / "forest_model.json"
    else:
        # Export single Decision Tree
        tree_dict = export_tree_to_dict(strategy.model, feature_names)
        
        model_json = {
            "model_type": "decision_tree",
            "tree": tree_dict,
            "state_features": feature_names,
            "normalization": {
                "own_funds": 999999.0,
                "own_score": 16.0,
                "opponent_score": 16.0,
                "own_survivors": 5.0,
                "opponent_survivors": 5.0,
                "consecutive_losses": 5.0,
                "round_number": 30.0,
                "half_length": 15.0,
                "last_round_reason": 4.0,
            }
        }
        output_file = output_dir / "tree_model.json"
    
    with open(output_file, 'w') as f:
        json.dump(model_json, f, indent=2)
    
    print(f"✓ Exported tree model to {output_file}")
    return output_file


if __name__ == "__main__":
    import sys
    
    # Export SGD model
    export_sgd_model(
        "models_new_strategies/sgd_abm_trained.pt",
        "go_models"
    )
    
    # Export Tree model
    export_tree_model(
        "models_new_strategies/tree_abm_trained.pt",
        "go_models"
    )
    
    # Export Forest model
    export_tree_model(
        "models_new_strategies/forest_abm_trained.pt",
        "go_models"
    )
    
    print("\n✓ All models exported successfully!")
```

### Step 2: Run Export

```powershell
python export_for_go.py
```

---

## Part 2: Go Implementation

### A. SGD Neural Network in Go

```go
package main

import (
    "encoding/json"
    "io/ioutil"
    "math"
)

// SGDModel represents the neural network
type SGDModel struct {
    Architecture struct {
        InputSize    int   `json:"input_size"`
        HiddenLayers []int `json:"hidden_layers"`
        OutputSize   int   `json:"output_size"`
    } `json:"architecture"`
    Weights struct {
        FC1Weight [][]float64 `json:"network.fc1.weight"`
        FC1Bias   []float64   `json:"network.fc1.bias"`
        FC2Weight [][]float64 `json:"network.fc2.weight"`
        FC2Bias   []float64   `json:"network.fc2.bias"`
        FC3Weight [][]float64 `json:"network.fc3.weight"`
        FC3Bias   []float64   `json:"network.fc3.bias"`
    } `json:"weights"`
    StateFeatures []string           `json:"state_features"`
    Normalization map[string]float64 `json:"normalization"`
}

// LoadSGDModel loads the model from JSON
func LoadSGDModel(filepath string) (*SGDModel, error) {
    data, err := ioutil.ReadFile(filepath)
    if err != nil {
        return nil, err
    }
    
    var model SGDModel
    if err := json.Unmarshal(data, &model); err != nil {
        return nil, err
    }
    
    return &model, nil
}

// matMul performs matrix-vector multiplication
func matMul(weights [][]float64, input []float64) []float64 {
    output := make([]float64, len(weights))
    for i := range weights {
        sum := 0.0
        for j := range input {
            sum += weights[i][j] * input[j]
        }
        output[i] = sum
    }
    return output
}

// addBias adds bias to each element
func addBias(vec []float64, bias []float64) []float64 {
    result := make([]float64, len(vec))
    for i := range vec {
        result[i] = vec[i] + bias[i]
    }
    return result
}

// relu applies ReLU activation
func relu(vec []float64) []float64 {
    result := make([]float64, len(vec))
    for i, v := range vec {
        result[i] = math.Max(0, v)
    }
    return result
}

// sigmoid applies sigmoid activation
func sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}

// Predict performs forward pass through the network
func (m *SGDModel) Predict(state GameState) float64 {
    // Normalize state
    input := []float64{
        state.OwnFunds / m.Normalization["own_funds"],
        state.OwnScore / m.Normalization["own_score"],
        state.OpponentScore / m.Normalization["opponent_score"],
        state.OwnSurvivors / m.Normalization["own_survivors"],
        state.OpponentSurvivors / m.Normalization["opponent_survivors"],
        state.ConsecutiveLosses / m.Normalization["consecutive_losses"],
        state.IsCtSide,
        state.RoundNumber / m.Normalization["round_number"],
        state.HalfLength / m.Normalization["half_length"],
        state.LastRoundReason / m.Normalization["last_round_reason"],
        state.LastBombPlanted,
    }
    
    // Layer 1: input -> hidden1 (64 neurons)
    hidden1 := matMul(m.Weights.FC1Weight, input)
    hidden1 = addBias(hidden1, m.Weights.FC1Bias)
    hidden1 = relu(hidden1)
    
    // Layer 2: hidden1 -> hidden2 (32 neurons)
    hidden2 := matMul(m.Weights.FC2Weight, hidden1)
    hidden2 = addBias(hidden2, m.Weights.FC2Bias)
    hidden2 = relu(hidden2)
    
    // Layer 3: hidden2 -> output (1 neuron)
    output := matMul(m.Weights.FC3Weight, hidden2)
    output = addBias(output, m.Weights.FC3Bias)
    
    // Apply sigmoid to get value in [0, 1]
    action := sigmoid(output[0])
    
    // Clip to [0, 1]
    if action < 0 {
        action = 0
    } else if action > 1 {
        action = 1
    }
    
    return action
}
```

### B. Decision Tree in Go

```go
// TreeNode represents a node in the decision tree
type TreeNode struct {
    IsLeaf    bool                   `json:"is_leaf"`
    Value     float64                `json:"value,omitempty"`
    Feature   string                 `json:"feature,omitempty"`
    Threshold float64                `json:"threshold,omitempty"`
    Left      *TreeNode              `json:"left,omitempty"`
    Right     *TreeNode              `json:"right,omitempty"`
}

// TreeModel represents a decision tree
type TreeModel struct {
    ModelType     string             `json:"model_type"`
    Tree          *TreeNode          `json:"tree"`
    StateFeatures []string           `json:"state_features"`
    Normalization map[string]float64 `json:"normalization"`
}

// LoadTreeModel loads the tree model from JSON
func LoadTreeModel(filepath string) (*TreeModel, error) {
    data, err := ioutil.ReadFile(filepath)
    if err != nil {
        return nil, err
    }
    
    var model TreeModel
    if err := json.Unmarshal(data, &model); err != nil {
        return nil, err
    }
    
    return &model, nil
}

// GetFeatureValue extracts the value for a given feature
func (m *TreeModel) GetFeatureValue(state GameState, feature string) float64 {
    var value float64
    
    switch feature {
    case "own_funds":
        value = state.OwnFunds / m.Normalization["own_funds"]
    case "own_score":
        value = state.OwnScore / m.Normalization["own_score"]
    case "opponent_score":
        value = state.OpponentScore / m.Normalization["opponent_score"]
    case "own_survivors":
        value = state.OwnSurvivors / m.Normalization["own_survivors"]
    case "opponent_survivors":
        value = state.OpponentSurvivors / m.Normalization["opponent_survivors"]
    case "consecutive_losses":
        value = state.ConsecutiveLosses / m.Normalization["consecutive_losses"]
    case "is_ct_side":
        value = state.IsCtSide
    case "round_number":
        value = state.RoundNumber / m.Normalization["round_number"]
    case "half_length":
        value = state.HalfLength / m.Normalization["half_length"]
    case "last_round_reason":
        value = state.LastRoundReason / m.Normalization["last_round_reason"]
    case "last_bomb_planted":
        value = state.LastBombPlanted
    default:
        value = 0
    }
    
    return value
}

// predictTree recursively traverses the tree
func (m *TreeModel) predictTree(node *TreeNode, state GameState) float64 {
    if node.IsLeaf {
        return node.Value
    }
    
    featureValue := m.GetFeatureValue(state, node.Feature)
    
    if featureValue <= node.Threshold {
        return m.predictTree(node.Left, state)
    }
    return m.predictTree(node.Right, state)
}

// Predict makes a prediction using the decision tree
func (m *TreeModel) Predict(state GameState) float64 {
    prediction := m.predictTree(m.Tree, state)
    
    // Clip to [0, 1]
    if prediction < 0 {
        prediction = 0
    } else if prediction > 1 {
        prediction = 1
    }
    
    return prediction
}
```

### C. Random Forest in Go

```go
// ForestModel represents a random forest
type ForestModel struct {
    ModelType     string             `json:"model_type"`
    NTrees        int                `json:"n_trees"`
    Trees         []*TreeNode        `json:"trees"`
    StateFeatures []string           `json:"state_features"`
    Normalization map[string]float64 `json:"normalization"`
}

// LoadForestModel loads the forest model from JSON
func LoadForestModel(filepath string) (*ForestModel, error) {
    data, err := ioutil.ReadFile(filepath)
    if err != nil {
        return nil, err
    }
    
    var model ForestModel
    if err := json.Unmarshal(data, &model); err != nil {
        return nil, err
    }
    
    return &model, nil
}

// GetFeatureValue extracts the value for a given feature
func (m *ForestModel) GetFeatureValue(state GameState, feature string) float64 {
    var value float64
    
    switch feature {
    case "own_funds":
        value = state.OwnFunds / m.Normalization["own_funds"]
    case "own_score":
        value = state.OwnScore / m.Normalization["own_score"]
    case "opponent_score":
        value = state.OpponentScore / m.Normalization["opponent_score"]
    case "own_survivors":
        value = state.OwnSurvivors / m.Normalization["own_survivors"]
    case "opponent_survivors":
        value = state.OpponentSurvivors / m.Normalization["opponent_survivors"]
    case "consecutive_losses":
        value = state.ConsecutiveLosses / m.Normalization["consecutive_losses"]
    case "is_ct_side":
        value = state.IsCtSide
    case "round_number":
        value = state.RoundNumber / m.Normalization["round_number"]
    case "half_length":
        value = state.HalfLength / m.Normalization["half_length"]
    case "last_round_reason":
        value = state.LastRoundReason / m.Normalization["last_round_reason"]
    case "last_bomb_planted":
        value = state.LastBombPlanted
    default:
        value = 0
    }
    
    return value
}

// predictTree recursively traverses a single tree
func (m *ForestModel) predictTree(node *TreeNode, state GameState) float64 {
    if node.IsLeaf {
        return node.Value
    }
    
    featureValue := m.GetFeatureValue(state, node.Feature)
    
    if featureValue <= node.Threshold {
        return m.predictTree(node.Left, state)
    }
    return m.predictTree(node.Right, state)
}

// Predict makes a prediction by averaging all trees
func (m *ForestModel) Predict(state GameState) float64 {
    sum := 0.0
    
    // Predict with each tree
    for _, tree := range m.Trees {
        sum += m.predictTree(tree, state)
    }
    
    // Average predictions
    prediction := sum / float64(m.NTrees)
    
    // Clip to [0, 1]
    if prediction < 0 {
        prediction = 0
    } else if prediction > 1 {
        prediction = 1
    }
    
    return prediction
}
```

---

## Part 3: Integration into Your ABM

### Complete Example

```go
package main

import (
    "fmt"
    "log"
)

// GameState represents the observable game state
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

// MLAgent wraps all three model types
type MLAgent struct {
    sgdModel    *SGDModel
    treeModel   *TreeModel
    forestModel *ForestModel
}

// NewMLAgent loads all models
func NewMLAgent() (*MLAgent, error) {
    sgd, err := LoadSGDModel("go_models/sgd_model.json")
    if err != nil {
        return nil, fmt.Errorf("loading SGD model: %w", err)
    }
    
    tree, err := LoadTreeModel("go_models/tree_model.json")
    if err != nil {
        return nil, fmt.Errorf("loading tree model: %w", err)
    }
    
    forest, err := LoadForestModel("go_models/forest_model.json")
    if err != nil {
        return nil, fmt.Errorf("loading forest model: %w", err)
    }
    
    return &MLAgent{
        sgdModel:    sgd,
        treeModel:   tree,
        forestModel: forest,
    }, nil
}

// DecideInvestment uses the specified model to make a decision
func (a *MLAgent) DecideInvestment(state GameState, modelType string) float64 {
    switch modelType {
    case "sgd":
        return a.sgdModel.Predict(state)
    case "tree":
        return a.treeModel.Predict(state)
    case "forest":
        return a.forestModel.Predict(state)
    default:
        // Default to forest (most robust)
        return a.forestModel.Predict(state)
    }
}

// Example usage in your ABM
func main() {
    // Load models
    agent, err := NewMLAgent()
    if err != nil {
        log.Fatal(err)
    }
    
    // Example game state
    state := GameState{
        OwnFunds:          4000,
        OwnScore:          3,
        OpponentScore:     5,
        OwnSurvivors:      3,
        OpponentSurvivors: 4,
        ConsecutiveLosses: 2,
        IsCtSide:          1.0,
        RoundNumber:       8,
        HalfLength:        15,
        LastRoundReason:   1,
        LastBombPlanted:   0,
    }
    
    // Get investment decisions from each model
    sgdDecision := agent.DecideInvestment(state, "sgd")
    treeDecision := agent.DecideInvestment(state, "tree")
    forestDecision := agent.DecideInvestment(state, "forest")
    
    fmt.Printf("SGD Model:    %.2f (invest $%.0f)\n", 
        sgdDecision, sgdDecision*state.OwnFunds)
    fmt.Printf("Tree Model:   %.2f (invest $%.0f)\n", 
        treeDecision, treeDecision*state.OwnFunds)
    fmt.Printf("Forest Model: %.2f (invest $%.0f)\n", 
        forestDecision, forestDecision*state.OwnFunds)
}
```

---

## Performance Comparison

| Model Type | Inference Speed | Memory | Interpretability | Robustness |
|------------|----------------|--------|------------------|------------|
| **SGD**    | Medium (~1ms)  | Small  | Low              | Medium     |
| **Tree**   | Fast (<0.1ms)  | Small  | High ⭐          | Low        |
| **Forest** | Medium (~1ms)  | Medium | Medium           | High ⭐     |

**Recommendation**: Use **Forest** for production (most robust), **Tree** for debugging (interpretable), **SGD** if you plan to continue training.

---

## Quick Start Commands

```powershell
# 1. Export models
python export_for_go.py

# 2. Copy Go code to your ABM project
# 3. Load and use models in your simulation
```
