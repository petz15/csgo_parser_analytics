"""
Train RL models using actual ABM simulation data

This script learns from historical ABM simulation results to understand
optimal economic strategies based on real game outcomes.
"""

import json
import os
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict
import torch

from ml_model import (
    GameState, ForbiddenGameState, Experience, BaseStrategy, StrategyFactory,
    DQNStrategy, PPOStrategy, REINFORCEStrategy, save_training_results
)


class ABMDataLoader:
    """
    Load and process ABM simulation data for training
    """
    
    def __init__(self, results_folder: str, use_forbidden_state: bool = False):
        self.results_folder = Path(results_folder)
        self.summary = None
        self.sim_files = []
        self.use_forbidden_state = use_forbidden_state
        
        self._discover_files()
        
        if use_forbidden_state:
            print("WARNING: Using ForbiddenGameState with opponent's private information!")
    
    def _discover_files(self):
        """Find all simulation CSV files and load summary"""
        if not self.results_folder.exists():
            raise FileNotFoundError(f"Results folder not found: {self.results_folder}")

        # Load summary
        summary_path = self.results_folder / 'simulation_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
            print(f"\n📊 Loaded ABM Summary:")
            print(f"  Total simulations: {self.summary.get('total_simulations','?')}")
            print(f"  Total rounds: {self.summary.get('total_rounds','?')}")
            print(f"  Team1 win rate: {self.summary.get('team1_win_rate',0):.2f}%")
            print(f"  Team2 win rate: {self.summary.get('team2_win_rate',0):.2f}%")
            print(f"  Average rounds per game: {self.summary.get('average_rounds',0):.2f}")

        # Find all simulation CSV files
        self.sim_files = sorted(list(self.results_folder.glob('*.csv')))
        print(f"  Found {len(self.sim_files)} simulation CSV files")
    
    def load_simulation(self, file_path: Path) -> Tuple[list, list]:
        """Load a single simulation CSV file and return rows and headers"""
        import csv
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            rows = list(reader)
        if not rows:
            return [], []
        headers = rows[0]
        data_rows = rows[1:]
        return data_rows, headers
    
    def extract_trajectories(self, data_rows, headers, use_forbidden_state: bool = None) -> Tuple[List[Experience], List[Experience]]:
        """
        Extract experience trajectories for both teams from CSV rows and headers
        Returns: (team1_experiences, team2_experiences)
        
        Args:
            data_rows: CSV data rows
            headers: CSV headers
            use_forbidden_state: If True, create ForbiddenGameState with opponent info. 
                               If None, uses self.use_forbidden_state
        """
        if use_forbidden_state is None:
            use_forbidden_state = self.use_forbidden_state
        # Get half length and OT info from summary
        half_length = 15
        ot_half_length = 3
        if self.summary:
            half_length = self.summary.get('GameRules', {}).get('halfLength', 15)
            ot_half_length = self.summary.get('GameRules', {}).get('otHalfLength', 3)

        # Map headers to indices
        header_idx = {h: i for i, h in enumerate(headers)}

        # Typical column names (customize if needed)
        t1_prefix = 't1_'
        t2_prefix = 't2_'

        # Required columns
        def col(name, team):
            prefix = t1_prefix if team == 1 else t2_prefix
            return header_idx.get(prefix + name)

        # Experience lists
        team1_experiences = []
        team2_experiences = []

        prev_row = None
        for i, row in enumerate(data_rows):
            # Parse round number
            round_num = int(row[header_idx.get('round_number', 0)])

            # Team sides (CT/T)
            team1_is_ct = row[header_idx.get('is_t1_ct', -1)] == 'true' if header_idx.get('is_t1_ct', -1) != -1 else True
            team2_is_ct = not team1_is_ct  # T2 is opposite of T1

            # Scores
            team1_score = int(row[col('score_end', 1)]) if col('score_end', 1) is not None else 0
            team2_score = int(row[col('score_end', 2)]) if col('score_end', 2) is not None else 0

            # Survivors
            team1_survivors = int(row[col('survivors', 1)]) if col('survivors', 1) is not None else 5
            team2_survivors = int(row[col('survivors', 2)]) if col('survivors', 2) is not None else 5

            # Consecutive losses
            team1_consec_losses = int(row[col('consecutive_losses', 1)]) if col('consecutive_losses', 1) is not None else 0
            team2_consec_losses = int(row[col('consecutive_losses', 2)]) if col('consecutive_losses', 2) is not None else 0

            # Funds
            team1_funds = float(row[col('funds_start', 1)]) if col('funds_start', 1) is not None else 4000
            team2_funds = float(row[col('funds_start', 2)]) if col('funds_start', 2) is not None else 4000

            # Equipment value
            team1_equip = float(row[col('fte_eq', 1)]) if col('fte_eq', 1) is not None else 0
            team2_equip = float(row[col('fte_eq', 2)]) if col('fte_eq', 2) is not None else 0

            # starting Equipment value
            team1_rs_equip = float(row[col('rs_eq', 1)]) if col('rs_eq', 1) is not None else 0
            team2_rs_equip = float(row[col('rs_eq', 2)]) if col('rs_eq', 2) is not None else 0

            # Earned (reward)
            team1_earned = float(row[col('earned', 1)]) if col('earned', 1) is not None else 0
            team2_earned = float(row[col('earned', 2)]) if col('earned', 2) is not None else 0

            #spent calculation
            team1_spent = float(row[col('spent', 1)]) if col('spent', 1) is not None else 0
            team2_spent = float(row[col('spent', 2)]) if col('spent', 2) is not None else 0

            # Compute spend ratio and investment ratio
            # Spend ratio: money_spent / money_start (how much of available funds were spent)
            # Investment ratio: fte_equipment / (money_start + rs_equipment) (equipment value relative to total resources)
            
            # Spend ratio = fte_eq / funds_start
            spend_ratio1 = min(1.0, team1_spent / max(team1_funds, 1))
            spend_ratio2 = min(1.0, team2_spent / max(team2_funds, 1))
            
            # Investment ratio = fte_eq / (funds_start + rs_eq)
            investment_ratio1 = min(1.0, team1_equip / max(team1_funds + team1_rs_equip, 1))
            investment_ratio2 = min(1.0, team2_equip / max(team2_funds + team2_rs_equip, 1))
            
            # Use investment ratio as the action (models equipment value relative to total resources)
            action1 = investment_ratio1
            action2 = investment_ratio2

            # Calculate last_round_reason and last_bomb_planted
            if prev_row is not None:
                last_reason = int(prev_row[header_idx.get('outcome_reason_code', -1)]) if header_idx.get('outcome_reason_code', -1) != -1 else 0
                last_bomb = prev_row[header_idx.get('outcome_bomb_planted', -1)] == 'true' if header_idx.get('outcome_bomb_planted', -1) != -1 else False
                team1_prev_survivors = int(prev_row[col('survivors', 1)]) if col('survivors', 1) is not None else 5
                team2_prev_survivors = int(prev_row[col('survivors', 2)]) if col('survivors', 2) is not None else 5
            else:
                last_reason = 0
                last_bomb = False
                team1_prev_survivors = 5
                team2_prev_survivors = 5

            # Create GameState or ForbiddenGameState for each team
            StateClass = ForbiddenGameState if use_forbidden_state else GameState
            
            if use_forbidden_state:
                # ForbiddenGameState with opponent's private information
                state1 = ForbiddenGameState(
                    own_funds=team1_funds,
                    own_score=team1_score,
                    opponent_score=team2_score,
                    own_survivors=team1_prev_survivors,
                    opponent_survivors=team2_prev_survivors,
                    consecutive_losses=team1_consec_losses,
                    is_ct_side=team1_is_ct,
                    round_number=round_num,
                    half_length=half_length,
                    last_round_reason=last_reason,
                    last_bomb_planted=last_bomb,
                    own_starting_equipment=team1_rs_equip,
                    opponent_funds=team2_funds,  # FORBIDDEN
                    opponent_starting_equipment=team2_rs_equip  # FORBIDDEN
                )
                state2 = ForbiddenGameState(
                    own_funds=team2_funds,
                    own_score=team2_score,
                    opponent_score=team1_score,
                    own_survivors=team2_prev_survivors,
                    opponent_survivors=team1_prev_survivors,
                    consecutive_losses=team2_consec_losses,
                    is_ct_side=team2_is_ct,
                    round_number=round_num,
                    half_length=half_length,
                    last_round_reason=last_reason,
                    last_bomb_planted=last_bomb,
                    own_starting_equipment=team2_rs_equip,
                    opponent_funds=team1_funds,  # FORBIDDEN
                    opponent_starting_equipment=team1_rs_equip  # FORBIDDEN
                )
            else:
                # Regular GameState (observable information only)
                state1 = GameState(
                    own_funds=team1_funds,
                    own_score=team1_score,
                    opponent_score=team2_score,
                    own_survivors=team1_prev_survivors,
                    opponent_survivors=team2_prev_survivors,
                    consecutive_losses=team1_consec_losses,
                    is_ct_side=team1_is_ct,
                    round_number=round_num,
                    half_length=half_length,
                    last_round_reason=last_reason,
                    last_bomb_planted=last_bomb,
                    own_starting_equipment=team1_rs_equip
                )
                state2 = GameState(
                    own_funds=team2_funds,
                    own_score=team2_score,
                    opponent_score=team1_score,
                    own_survivors=team2_prev_survivors,
                    opponent_survivors=team1_prev_survivors,
                    consecutive_losses=team2_consec_losses,
                    is_ct_side=team2_is_ct,
                    round_number=round_num,
                    half_length=half_length,
                    last_round_reason=last_reason,
                    last_bomb_planted=last_bomb,
                    own_starting_equipment=team2_rs_equip
                )

            # Reward for each team
            round_reward1 = team1_earned
            round_reward2 = team2_earned

            # Next state (filled in next iteration)
            next_state1 = None
            next_state2 = None

            team1_experiences.append((state1, action1, round_reward1))
            team2_experiences.append((state2, action2, round_reward2))

            prev_row = row

        # Convert to Experience objects
        team1_exp_list = []
        for i, (state, action, reward) in enumerate(team1_experiences):
            next_state = team1_experiences[i + 1][0] if i < len(team1_experiences) - 1 else None
            done = (i == len(team1_experiences) - 1)
            team1_exp_list.append(Experience(state, action, reward, next_state, done))

        team2_exp_list = []
        for i, (state, action, reward) in enumerate(team2_experiences):
            next_state = team2_experiences[i + 1][0] if i < len(team2_experiences) - 1 else None
            done = (i == len(team2_experiences) - 1)
            team2_exp_list.append(Experience(state, action, reward, next_state, done))

        return team1_exp_list, team2_exp_list


class ImitationTrainer:
    """
    Train strategies using imitation learning from ABM data
    """
    
    def __init__(self, data_loader: ABMDataLoader):
        self.data_loader = data_loader
        self.experiences = defaultdict(list)
    
    def collect_experiences(self, n_simulations: int = None):
        """
        Collect experiences from simulation CSV files
        """
        sim_files = self.data_loader.sim_files
        if n_simulations:
            sim_files = sim_files[:n_simulations]

        print(f"\n📥 Collecting experiences from {len(sim_files)} simulations...")

        total_rounds = 0
        team1_experiences = []
        team2_experiences = []

        for sim_file in tqdm(sim_files, desc="Loading simulations"):
            try:
                data_rows, headers = self.data_loader.load_simulation(sim_file)
                t1_exp, t2_exp = self.data_loader.extract_trajectories(data_rows, headers)
                team1_experiences.extend(t1_exp)
                team2_experiences.extend(t2_exp)
                total_rounds += len(t1_exp)
            except Exception as e:
                print(f"Error processing {sim_file.name}: {e}")
                continue

        print(f"\n✓ Collected {total_rounds} rounds of experience")
        print(f"  Team1 experiences: {len(team1_experiences)}")
        print(f"  Team2 experiences: {len(team2_experiences)}")

        return team1_experiences, team2_experiences
    
    def train_strategy(self, strategy: BaseStrategy, experiences: List[Experience],
                      n_epochs: int = 10, batch_size: int = 64):
        """
        Train a strategy using collected experiences
        """
        if not experiences:
            print("No experiences to train on!")
            return
        
        # Only RL strategies can be trained
        if not hasattr(strategy, 'update'):
            print(f"Strategy {strategy.name} cannot be trained (no update method)")
            return
        
        # Import for type checking
        from ml_model import DQNStrategy, PPOStrategy, REINFORCEStrategy, SGDStrategy, TreeStrategy, XGBoostStrategy, LogisticStrategy
        
        print(f"\n🎯 Training {strategy.name}...")
        print(f"  Total experiences: {len(experiences)}")
        print(f"  Epochs: {n_epochs}")
        
        # For Tree-based, XGBoost, and Logistic strategies, train directly on all data
        if isinstance(strategy, (TreeStrategy, XGBoostStrategy, LogisticStrategy)):
            print(f"  {strategy.name} learning: feeding all experiences...")
            for exp in tqdm(experiences, desc="Adding experiences"):
                strategy.update(exp)
            
            # Final training
            print(f"  Finalizing {strategy.name} training...")
            strategy.finalize_training()
            print(f"  {strategy.name} model trained on {len(experiences)} experiences")
            return
        
        # For SGD strategy, incremental learning
        if isinstance(strategy, SGDStrategy):
            print("  Incremental learning with SGD...")
            for epoch in range(n_epochs):
                np.random.shuffle(experiences)
                losses = []
                
                for exp in tqdm(experiences, desc=f"Epoch {epoch+1}/{n_epochs}"):
                    strategy.update(exp)
                    if strategy.losses:
                        losses.append(strategy.losses[-1])
                
                if losses:
                    print(f"  Epoch {epoch+1} avg loss: {np.mean(losses[-100:]):.4f}")
            return
        
        # For DQN, add experiences to replay buffer
        if isinstance(strategy, DQNStrategy):
            for exp in tqdm(experiences, desc="Adding to replay buffer"):
                strategy.memory.append(exp)
            
            # Train
            n_updates = len(experiences) // batch_size
            for epoch in range(n_epochs):
                losses = []
                for _ in tqdm(range(n_updates), desc=f"Epoch {epoch+1}/{n_epochs}"):
                    strategy._train_step()
                    if strategy.losses:
                        losses.append(strategy.losses[-1])
                
                if losses:
                    print(f"  Epoch {epoch+1} avg loss: {np.mean(losses):.4f}")
        
        # For PPO/REINFORCE, need to restructure training
        elif isinstance(strategy, (PPOStrategy, REINFORCEStrategy)):
            # These methods expect episode-based training
            # Group experiences by episodes (matches)
            print("  Episode-based training for policy gradient methods...")
            
            # Process in batches
            for epoch in range(n_epochs):
                np.random.shuffle(experiences)
                
                for i in tqdm(range(0, len(experiences), batch_size), desc=f"Epoch {epoch+1}/{n_epochs}"):
                    batch = experiences[i:i+batch_size]
                    
                    # Add to strategy buffer
                    for exp in batch:
                        if isinstance(strategy, PPOStrategy):
                            strategy.states.append(exp.state.to_array())
                            strategy.actions.append(exp.action)
                            strategy.rewards.append(exp.reward)
                            strategy.values.append(0.0)  # Placeholder
                            strategy.log_probs.append(0.0)  # Placeholder
                            strategy.dones.append(exp.done)
                        else:  # REINFORCE
                            strategy.states.append(exp.state.to_array())
                            strategy.actions.append(exp.action)
                            strategy.rewards.append(exp.reward)
                    
                    # Update
                    if len(strategy.states) >= batch_size:
                        strategy.update()
                
                print(f"  Epoch {epoch+1} completed")


def main():
    parser = argparse.ArgumentParser(description='Train RL models from ABM simulation data')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing ABM simulation results')
    parser.add_argument('--n-simulations', type=int, default=None,
                       help='Number of simulations to load (default: all)')
    parser.add_argument('--strategies', nargs='+', default=['dqn', 'ppo'],
                       help='Strategies to train')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--save-dir', type=str, default='models_from_abm',
                       help='Directory to save trained models')
    parser.add_argument('--use-winning-only', action='store_true',
                       help='Train only on winning team behaviors')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Computation device: auto (default), cuda, or cpu')
    parser.add_argument('--export-go-dir', type=str, default=None,
                       help='If provided, export trained models to Go JSON in this directory after training')
    parser.add_argument('--use-forbidden-state', action='store_true',
                       help='Use ForbiddenGameState with opponent private information')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CS:GO Economic ABM - Training from Simulation Data")
    print("=" * 70)
    
    # Load ABM data
    data_loader = ABMDataLoader(args.results_dir, use_forbidden_state=args.use_forbidden_state)
    
    # Collect experiences
    trainer = ImitationTrainer(data_loader)
    winning_exp, losing_exp = trainer.collect_experiences(args.n_simulations)
    
    # Choose which experiences to use
    if args.use_winning_only:
        print("\n🏆 Training on winning team behaviors only")
        training_experiences = winning_exp
    else:
        print("\n⚖️ Training on all team behaviors (winning + losing)")
        training_experiences = winning_exp + losing_exp
    
    # Create strategies
    # Resolve device
    if args.device == 'auto':
        resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available. Install CUDA-capable PyTorch or select cpu.')
        resolved_device = args.device

    print(f"\n🤖 Initializing strategies: {', '.join(args.strategies)}")
    print(f"  Using device: {resolved_device}")
    strategies = {}
    for strategy_type in args.strategies:
        try:
            strategies[strategy_type] = StrategyFactory.create(strategy_type)
            print(f"  ✓ {strategy_type}")
        except Exception as e:
            print(f"  ✗ {strategy_type}: {e}")
    
    if not strategies:
        print("No valid strategies to train!")
        return
    
    # Helper to move strategy modules to target device (for explicit override)
    def move_strategy_to_device(strategy, device_str: str):
        if not hasattr(strategy, 'device'):
            return
        device = torch.device(device_str)
        strategy.device = device
        # DQN
        from ml_model import DQNStrategy, PPOStrategy, REINFORCEStrategy, SGDStrategy
        if isinstance(strategy, DQNStrategy):
            strategy.q_network.to(device)
            strategy.target_network.to(device)
        elif isinstance(strategy, PPOStrategy):
            strategy.policy.to(device)
            strategy.value.to(device)
        elif isinstance(strategy, REINFORCEStrategy):
            strategy.policy.to(device)
        elif isinstance(strategy, SGDStrategy):
            strategy.model.to(device)

    # Explicitly move strategies if user forced device (when auto we rely on internal init)
    if args.device != 'auto':
        for s in strategies.values():
            move_strategy_to_device(s, resolved_device)

    # Train each strategy
    for name, strategy in strategies.items():
        try:
            trainer.train_strategy(
                strategy,
                training_experiences,
                n_epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            # Save model (add 'forbidden' suffix if using forbidden state)
            os.makedirs(args.save_dir, exist_ok=True)
            suffix = "_forbidden" if args.use_forbidden_state else ""
            save_path = os.path.join(args.save_dir, f"{name}_abm_trained{suffix}.pt")
            strategy.save(save_path)
            print(f"  ✓ Saved to {save_path}")
            
        except Exception as e:
            print(f"  ✗ Error training {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Models saved to: {args.save_dir}")

    # Optional Go export
    if args.export_go_dir:
        try:
            from export_models import export_all_models
            go_export_dir = args.export_go_dir
            print(f"\n🚀 Exporting models to Go format in: {go_export_dir}")
            export_all_models(args.save_dir, go_export_dir)
        except Exception as e:
            print(f"✗ Go export failed: {e}")
    print("=" * 70)
    
    # Save training info
    info = {
        "results_dir": args.results_dir,
        "n_simulations": len(data_loader.sim_files) if not args.n_simulations else args.n_simulations,
        "total_experiences": len(training_experiences),
        "winning_experiences": len(winning_exp),
        "losing_experiences": len(losing_exp),
        "use_winning_only": args.use_winning_only,
        "use_forbidden_state": args.use_forbidden_state,
        "state_dim": 15 if args.use_forbidden_state else 13,
        "strategies": list(strategies.keys()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    
    info_path = os.path.join(args.save_dir, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nTraining info saved to: {info_path}")


if __name__ == "__main__":
    main()
