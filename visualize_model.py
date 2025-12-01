import numpy as np
import matplotlib.pyplot as plt
from ml_model import StrategyFactory, GameState
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize ML models')
parser.add_argument('--model', type=str, default='both', 
                    choices=['tree', 'forest', 'both', 'prediction'],
                    help='Which model to visualize: tree, forest, both, or prediction')
parser.add_argument('--tree-path', type=str, default='models_tree_test_cuda/tree_abm_trained.pt',
                    help='Path to DecisionTree model')
parser.add_argument('--forest-path', type=str, default='models_tree_forest_test/forest_abm_trained.pt',
                    help='Path to RandomForest model')
args = parser.parse_args()

# Load models based on selection
models = {}
if args.model in ['tree', 'both', 'prediction']:
    models["DecisionTree"] = StrategyFactory.create("tree")
    models["DecisionTree"].load(args.tree_path)
if args.model in ['forest', 'both', 'prediction']:
    models["RandomForest"] = StrategyFactory.create("forest")
    models["RandomForest"].load(args.forest_path)

from sklearn.tree import plot_tree, export_text

# Visualize tree structures based on model selection
if args.model in ['tree', 'both']:
    # Access underlying sklearn models
    sk_tree = models["DecisionTree"].model
    
    # Visualize DecisionTree - entire tree with max_depth=None
    plt.figure(figsize=(40, 20))
    plot_tree(sk_tree, filled=True, feature_names=[f"f{i}" for i in range(sk_tree.n_features_in_)], 
              max_depth=None, fontsize=8, rounded=True, proportion=True)
    plt.title("DecisionTree Structure (Complete)", fontsize=16)
    plt.tight_layout()
    plt.savefig("decision_tree_complete.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print DecisionTree as text
    print("DecisionTree structure:")
    print(export_text(sk_tree))

if args.model in ['forest', 'both']:
    # Access underlying sklearn models
    sk_forest = models["RandomForest"].model
    
    # Visualize first tree in RandomForest - entire tree with max_depth=None
    plt.figure(figsize=(40, 20))
    plot_tree(sk_forest.estimators_[0], filled=True, feature_names=[f"f{i}" for i in range(sk_forest.n_features_in_)],
              max_depth=None, fontsize=8, rounded=True, proportion=True)
    plt.title("RandomForest - Tree 0 Structure (Complete)", fontsize=16)
    plt.tight_layout()
    plt.savefig("random_forest_tree0_complete.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print first RandomForest tree as text
    print("RandomForest Tree 0 structure:")
    print(export_text(sk_forest.estimators_[0]))

# Generate prediction comparison plot
if args.model == 'prediction' or (args.model == 'both' and len(models) > 0):
    # Generate sample game states
    funds = np.linspace(1000, 20000, 50)
    actions = {}

    for name, model in models.items():
        actions[name] = []
        for f in funds:
            state = GameState(
                own_funds=f, own_score=10, opponent_score=10,
                own_survivors=5, opponent_survivors=5,
                consecutive_losses=0, is_ct_side=True,
                round_number=10, half_length=15,
                last_round_reason=1, last_bomb_planted=False
            )
            action = model.select_action(state, training=False)
            actions[name].append(action)

    # Plot
    plt.figure(figsize=(10, 6))
    for name, acts in actions.items():
        plt.plot(funds, acts, label=name)
    plt.xlabel("Own Funds")
    plt.ylabel("Predicted Investment Ratio")
    plt.title("ML Model Investment Predictions")
    plt.legend()
    plt.grid(True)
    plt.show()