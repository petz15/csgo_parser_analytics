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

from ml_model import DQNStrategy, PPOStrategy, REINFORCEStrategy, TreeStrategy

FEATURE_NAMES = [
    "own_funds", "own_score", "opponent_score",
    "own_survivors", "opponent_survivors", "consecutive_losses",
    "is_ct_side", "round_number", "half_length",
    "last_round_reason", "last_bomb_planted"
]

NORMALIZATION = {
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
def export_tree_to_dict(tree, feature_names):
    """Recursively export sklearn tree to dictionary"""
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != -2 else "undefined" for i in tree_.feature]
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
    try:
        import joblib
        data = joblib.load(model_path)
    except Exception as e:
        print(f"Could not load model from {model_path}: {e}")
        return

    # Unwrap dictionary format saved by TreeStrategy.save
    if isinstance(data, dict):
        if 'model' not in data:
            print(f"Tree model file {model_path} does not contain a trained model ('model' key missing); skipping export.")
            return None
        model_obj = data['model']
        is_forest = data.get('use_forest', hasattr(model_obj, 'estimators_'))
    else:
        model_obj = data
        is_forest = hasattr(model_obj, 'estimators_')

    # Basic validation
    if not hasattr(model_obj, 'tree_') and not hasattr(model_obj, 'estimators_'):
        print(f"Unsupported tree model type: {type(model_obj)}")
        return

    # Export
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if is_forest:
        trees = [export_tree_to_dict(est, FEATURE_NAMES) for est in model_obj.estimators_]
        model_json = {
            "model_type": "random_forest",
            "n_trees": len(trees),
            "trees": trees,
            "state_features": FEATURE_NAMES,
            "normalization": NORMALIZATION
        }
        output_file = output_dir / "forest_model.json"
    else:
        tree_dict = export_tree_to_dict(model_obj, FEATURE_NAMES)
        model_json = {
            "model_type": "decision_tree",
            "tree": tree_dict,
            "state_features": FEATURE_NAMES,
            "normalization": NORMALIZATION
        }
        output_file = output_dir / "tree_model.json"

    with open(output_file, 'w') as f:
        json.dump(model_json, f, indent=2)
    print(f"✓ Exported tree model to {output_file}")
    return output_file


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
    
    # (Optional) Inference template generation removed for simplicity


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


def export_reinforce_strategy(strategy: REINFORCEStrategy, output_dir: str):
    """Export REINFORCE strategy similar to PPO."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    policy_path = output_dir / "policy_network_weights.json"
    export_pytorch_model_to_json(strategy.policy, str(policy_path))

    metadata = {
        "model_type": "REINFORCE",
        "state_dim": strategy.state_dim,
        "architecture": {
            "layers": []
        },
        "win_rate": strategy.get_win_rate(),
        "total_matches": len(strategy.match_history),
    }
    for name, module in strategy.policy.network.named_children():
        if isinstance(module, torch.nn.Linear):
            metadata["architecture"]["layers"].append({
                "type": "linear",
                "in_features": module.in_features,
                "out_features": module.out_features,
            })
        elif isinstance(module, torch.nn.ReLU):
            metadata["architecture"]["layers"].append({"type": "relu"})
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported REINFORCE metadata to {metadata_path}")
    





def export_all_models(models_dir: str, output_base_dir: str):
    """
    Export all trained models in a directory
    """
    models_dir = Path(models_dir)
    output_base_dir = Path(output_base_dir)
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return
    
    # Find all model files (both .pt and .joblib)
    model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.joblib"))
    
    if not model_files:
        print(f"No model files (.pt or .joblib) found in {models_dir}")
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
                export_reinforce_strategy(strategy, str(output_dir))
            elif 'tree' in model_name.lower() or 'forest' in model_name.lower():
                result = export_tree_model(str(model_file), str(output_dir))
                if result is None:
                    raise RuntimeError("Tree/forest model not exported (untrained or incompatible format)")
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
