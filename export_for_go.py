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
    
    # Extract weights from the Sequential model
    weights = {}
    for name, param in strategy.model.named_parameters():
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
    except Exception as e:
        print(f"Could not load model from {model_path}: {e}")
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
    print("Exporting ML models to Go-compatible format...")
    print("=" * 60)
    
    # Export SGD model
    print("\n1. Exporting SGD model...")
    export_sgd_model(
        "models_new_strategies/sgd_abm_trained.pt",
        "go_models"
    )
    
    # Export Tree model
    print("\n2. Exporting Tree model...")
    export_tree_model(
        "models_new_strategies/tree_abm_trained.pt",
        "go_models"
    )
    
    # Export Forest model
    print("\n3. Exporting Forest model...")
    export_tree_model(
        "models_new_strategies/forest_abm_trained.pt",
        "go_models"
    )
    
    print("\n" + "=" * 60)
    print("✓ All models exported successfully to go_models/")
    print("\nNext steps:")
    print("  1. See GO_INTEGRATION_GUIDE.md for complete Go implementation")
    print("  2. Copy the Go code into your ABM project")
    print("  3. Load models with NewMLAgent()")
