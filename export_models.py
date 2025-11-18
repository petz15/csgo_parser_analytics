"""
Export trained RL models to Go-compatible format

This script exports trained PyTorch models to simple JSON format
that can be easily implemented in Go, including:
- Network weights and biases
- Model architecture
- Inference code template
"""

import torch
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List

from ml_model import DQNStrategy, PPOStrategy, REINFORCEStrategy


def export_pytorch_model_to_json(model: torch.nn.Module, filepath: str):
    """
    Export PyTorch model weights to JSON format
    """
    state_dict = model.state_dict()
    
    json_weights = {}
    for key, value in state_dict.items():
        # Convert to numpy, then to list for JSON serialization
        json_weights[key] = value.cpu().numpy().tolist()
    
    with open(filepath, 'w') as f:
        json.dump(json_weights, f, indent=2)
    
    print(f"Exported model weights to {filepath}")


def export_dqn_strategy(strategy: DQNStrategy, output_dir: str):
    """
    Export DQN strategy to Go-compatible format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Export Q-network weights
    q_network_path = output_dir / "q_network_weights.json"
    export_pytorch_model_to_json(strategy.q_network, str(q_network_path))
    
    # Export metadata
    metadata = {
        "model_type": "DQN",
        "state_dim": strategy.state_dim,
        "n_actions": strategy.n_actions,
        "action_values": strategy.action_values.tolist(),
        "architecture": {
            "layers": []
        },
        "win_rate": strategy.get_win_rate(),
        "total_matches": len(strategy.match_history),
    }
    
    # Extract architecture
    for name, module in strategy.q_network.network.named_children():
        if isinstance(module, torch.nn.Linear):
            metadata["architecture"]["layers"].append({
                "type": "linear",
                "in_features": module.in_features,
                "out_features": module.out_features,
            })
        elif isinstance(module, torch.nn.ReLU):
            metadata["architecture"]["layers"].append({
                "type": "relu"
            })
        elif isinstance(module, torch.nn.LayerNorm):
            metadata["architecture"]["layers"].append({
                "type": "layernorm",
                "normalized_shape": module.normalized_shape[0]
            })
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Exported DQN metadata to {metadata_path}")
    
    # Generate Go inference code template
    generate_go_inference_code(metadata, output_dir / "inference.go")


def export_ppo_strategy(strategy: PPOStrategy, output_dir: str):
    """
    Export PPO strategy to Go-compatible format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Export policy network weights
    policy_path = output_dir / "policy_network_weights.json"
    export_pytorch_model_to_json(strategy.policy, str(policy_path))
    
    # Export metadata
    metadata = {
        "model_type": "PPO",
        "state_dim": strategy.state_dim,
        "architecture": {
            "shared_layers": [],
            "alpha_head": {"in_features": 64, "out_features": 1},
            "beta_head": {"in_features": 64, "out_features": 1},
        },
        "win_rate": strategy.get_win_rate(),
        "total_matches": len(strategy.match_history),
    }
    
    # Extract architecture
    for name, module in strategy.policy.shared_layers.named_children():
        if isinstance(module, torch.nn.Linear):
            metadata["architecture"]["shared_layers"].append({
                "type": "linear",
                "in_features": module.in_features,
                "out_features": module.out_features,
            })
        elif isinstance(module, torch.nn.ReLU):
            metadata["architecture"]["shared_layers"].append({
                "type": "relu"
            })
        elif isinstance(module, torch.nn.LayerNorm):
            metadata["architecture"]["shared_layers"].append({
                "type": "layernorm",
                "normalized_shape": module.normalized_shape[0]
            })
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Exported PPO metadata to {metadata_path}")
    
    # Generate Go inference code template
    generate_go_inference_code_ppo(metadata, output_dir / "inference.go")


def generate_go_inference_code(metadata: Dict, output_path: Path):
    """
    Generate Go code template for DQN inference
    """
    
    go_code = f"""package main

import (
	"encoding/json"
	"math"
	"os"
)

// GameState represents the observable game state
type GameState struct {{
	OwnFunds          float64
	OwnScore          int
	OpponentScore     int
	OwnSurvivors      int
	OpponentSurvivors int
	ConsecutiveLosses int
	IsCTSide          bool
	RoundNumber       int
	HalfLength        int
	LastRoundReason   int
	LastBombPlanted   bool
}}

// ToArray converts GameState to normalized feature array
func (s *GameState) ToArray() []float64 {{
	ctSide := 0.0
	if s.IsCTSide {{
		ctSide = 1.0
	}}
	bombPlanted := 0.0
	if s.LastBombPlanted {{
		bombPlanted = 1.0
	}}
	
	return []float64{{
		s.OwnFunds / 50000.0,
		float64(s.OwnScore) / 16.0,
		float64(s.OpponentScore) / 16.0,
		float64(s.OwnSurvivors) / 5.0,
		float64(s.OpponentSurvivors) / 5.0,
		math.Min(float64(s.ConsecutiveLosses), 5) / 5.0,
		ctSide,
		float64(s.RoundNumber) / 30.0,
		float64(s.HalfLength) / 15.0,
		float64(s.LastRoundReason) / 4.0,
		bombPlanted,
	}}
}}

// DQNModel represents a trained DQN model
type DQNModel struct {{
	StateDim     int         `json:"state_dim"`
	NActions     int         `json:"n_actions"`
	ActionValues []float64   `json:"action_values"`
	Weights      ModelWeights
}}

// ModelWeights holds the neural network weights
type ModelWeights struct {{
	Layers []LayerWeights
}}

// LayerWeights represents weights for one layer
type LayerWeights struct {{
	Weight [][]float64
	Bias   []float64
}}

// LoadModel loads a DQN model from JSON files
func LoadModel(metadataPath, weightsPath string) (*DQNModel, error) {{
	// Load metadata
	metadataFile, err := os.ReadFile(metadataPath)
	if err != nil {{
		return nil, err
	}}
	
	var model DQNModel
	if err := json.Unmarshal(metadataFile, &model); err != nil {{
		return nil, err
	}}
	
	// Load weights
	weightsFile, err := os.ReadFile(weightsPath)
	if err != nil {{
		return nil, err
	}}
	
	var rawWeights map[string]interface{{}}
	if err := json.Unmarshal(weightsFile, &rawWeights); err != nil {{
		return nil, err
	}}
	
	// Parse weights into layer structure
	// This is simplified - actual implementation would need to parse the layer structure
	
	return &model, nil
}}

// ReLU activation function
func relu(x float64) float64 {{
	if x > 0 {{
		return x
	}}
	return 0
}}

// LayerNorm normalization (simplified)
func layerNorm(x []float64) []float64 {{
	mean := 0.0
	for _, v := range x {{
		mean += v
	}}
	mean /= float64(len(x))
	
	variance := 0.0
	for _, v := range x {{
		variance += (v - mean) * (v - mean)
	}}
	variance /= float64(len(x))
	
	result := make([]float64, len(x))
	for i, v := range x {{
		result[i] = (v - mean) / math.Sqrt(variance+1e-5)
	}}
	return result
}}

// Forward pass through linear layer
func linearForward(input []float64, weight [][]float64, bias []float64) []float64 {{
	output := make([]float64, len(bias))
	for i := range output {{
		sum := bias[i]
		for j := range input {{
			sum += input[j] * weight[i][j]
		}}
		output[i] = sum
	}}
	return output
}}

// Predict Q-values for a given state
func (m *DQNModel) Predict(state GameState) []float64 {{
	x := state.ToArray()
	
	// Forward pass through network
	// This is simplified - actual implementation would iterate through layers
	for _, layer := range m.Weights.Layers {{
		x = linearForward(x, layer.Weight, layer.Bias)
		// Apply activation (ReLU) except for last layer
		for i := range x {{
			x[i] = relu(x[i])
		}}
		x = layerNorm(x)
	}}
	
	return x
}}

// SelectAction selects the best action based on Q-values
func (m *DQNModel) SelectAction(state GameState) float64 {{
	qValues := m.Predict(state)
	
	// Find action with maximum Q-value
	maxIdx := 0
	maxVal := qValues[0]
	for i := 1; i < len(qValues); i++ {{
		if qValues[i] > maxVal {{
			maxVal = qValues[i]
			maxIdx = i
		}}
	}}
	
	return m.ActionValues[maxIdx]
}}

func main() {{
	// Example usage
	model, err := LoadModel("metadata.json", "q_network_weights.json")
	if err != nil {{
		panic(err)
	}}
	
	state := GameState{{
		OwnFunds:          10000,
		OwnScore:          7,
		OpponentScore:     5,
		OwnSurvivors:      3,
		OpponentSurvivors: 2,
		ConsecutiveLosses: 1,
		IsCTSide:          true,
		RoundNumber:       12,
		HalfLength:        15,
		LastRoundReason:   4,
		LastBombPlanted:   false,
	}}
	
	action := model.SelectAction(state)
	println("Selected investment ratio:", action)
}}
"""
    
    with open(output_path, 'w') as f:
        f.write(go_code)
    
    print(f"Generated Go inference template at {output_path}")


def generate_go_inference_code_ppo(metadata: Dict, output_path: Path):
    """
    Generate Go code template for PPO inference
    """
    
    go_code = """package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
)

// GameState - same as DQN version above

// PPOModel represents a trained PPO model
type PPOModel struct {
	StateDim int          `json:"state_dim"`
	Weights  PolicyWeights
}

// PolicyWeights holds the policy network weights
type PolicyWeights struct {
	SharedLayers []LayerWeights
	AlphaHead    LayerWeights
	BetaHead     LayerWeights
}

// Softplus activation: log(1 + exp(x))
func softplus(x float64) float64 {
	if x > 20 {
		return x  // Numerical stability
	}
	return math.Log(1 + math.Exp(x))
}

// BetaDistribution samples from Beta distribution
type BetaDistribution struct {
	Alpha float64
	Beta  float64
}

// Sample from Beta distribution using rejection sampling (simplified)
func (b *BetaDistribution) Sample() float64 {
	// Simplified sampling - actual implementation would use proper Beta sampling
	// For Go implementation, consider using gonum.org/v1/gonum/stat/distuv
	
	// Mean of Beta distribution as approximation
	return b.Alpha / (b.Alpha + b.Beta)
}

// Predict returns Beta distribution parameters
func (m *PPOModel) Predict(state GameState) (float64, float64) {
	x := state.ToArray()
	
	// Forward through shared layers
	for _, layer := range m.Weights.SharedLayers {
		x = linearForward(x, layer.Weight, layer.Bias)
		for i := range x {
			x[i] = relu(x[i])
		}
		x = layerNorm(x)
	}
	
	// Alpha and Beta heads
	alphaOut := linearForward(x, m.Weights.AlphaHead.Weight, m.Weights.AlphaHead.Bias)
	betaOut := linearForward(x, m.Weights.BetaHead.Weight, m.Weights.BetaHead.Bias)
	
	alpha := softplus(alphaOut[0]) + 1.0
	beta := softplus(betaOut[0]) + 1.0
	
	return alpha, beta
}

// SelectAction selects action by sampling from policy
func (m *PPOModel) SelectAction(state GameState) float64 {
	alpha, beta := m.Predict(state)
	
	dist := BetaDistribution{Alpha: alpha, Beta: beta}
	return dist.Sample()
}

func main() {
	// Example usage - same as DQN version
}
"""
    
    with open(output_path, 'w') as f:
        f.write(go_code)
    
    print(f"Generated Go inference template (PPO) at {output_path}")


def export_all_models(models_dir: str, output_base_dir: str):
    """
    Export all trained models in a directory
    """
    models_dir = Path(models_dir)
    output_base_dir = Path(output_base_dir)
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return
    
    # Find all model files
    model_files = list(models_dir.glob("*.pt"))
    
    if not model_files:
        print(f"No .pt model files found in {models_dir}")
        return
    
    print(f"\nFound {len(model_files)} model files")
    print("=" * 70)
    
    for model_file in model_files:
        model_name = model_file.stem
        print(f"\nExporting {model_name}...")
        
        output_dir = output_base_dir / model_name
        
        try:
            # Determine model type from name
            if 'dqn' in model_name.lower():
                strategy = DQNStrategy()
                strategy.load(str(model_file))
                export_dqn_strategy(strategy, str(output_dir))
            elif 'ppo' in model_name.lower():
                strategy = PPOStrategy()
                strategy.load(str(model_file))
                export_ppo_strategy(strategy, str(output_dir))
            elif 'reinforce' in model_name.lower():
                strategy = REINFORCEStrategy()
                strategy.load(str(model_file))
                # REINFORCE uses same structure as PPO for export
                export_ppo_strategy(strategy, str(output_dir))
            else:
                print(f"  Unknown model type: {model_name}, skipping")
                continue
            
            print(f"  ✓ Exported successfully to {output_dir}")
            
        except Exception as e:
            print(f"  ✗ Error exporting {model_name}: {e}")
    
    print("\n" + "=" * 70)
    print("Export complete!")


def main():
    parser = argparse.ArgumentParser(description='Export trained models to Go format')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='exported_models',
                       help='Output directory for exported models')
    
    args = parser.parse_args()
    
    print("Model Export Tool")
    print("=" * 70)
    print(f"Models directory: {args.models_dir}")
    print(f"Output directory: {args.output_dir}")
    
    export_all_models(args.models_dir, args.output_dir)


if __name__ == "__main__":
    main()
