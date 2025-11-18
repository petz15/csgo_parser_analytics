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

from ml_model import (
    GameState, Experience, BaseStrategy, StrategyFactory,
    DQNStrategy, PPOStrategy, REINFORCEStrategy, save_training_results
)


class ABMDataLoader:
    """
    Load and process ABM simulation data for training
    """
    
    def __init__(self, results_folder: str):
        self.results_folder = Path(results_folder)
        self.summary = None
        self.sim_files = []
        
        self._discover_files()
    
    def _discover_files(self):
        """Find all simulation files"""
        if not self.results_folder.exists():
            raise FileNotFoundError(f"Results folder not found: {self.results_folder}")
        
        # Load summary
        summary_path = self.results_folder / 'simulation_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
            print(f"\n📊 Loaded ABM Summary:")
            print(f"  Total simulations: {self.summary['total_simulations']}")
            print(f"  Total rounds: {self.summary['total_rounds']}")
            print(f"  Team1 win rate: {self.summary['team1_win_rate']:.2f}%")
            print(f"  Team2 win rate: {self.summary['team2_win_rate']:.2f}%")
            print(f"  Average rounds per game: {self.summary['average_rounds']:.2f}")
        
        # Find all simulation files
        self.sim_files = sorted(list(self.results_folder.glob('sim_*.json')))
        print(f"  Found {len(self.sim_files)} simulation files")
    
    def load_simulation(self, file_path: Path) -> Dict:
        """Load a single simulation file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def extract_trajectories(self, simulation: Dict) -> Tuple[List[Experience], List[Experience]]:
        """
        Extract experience trajectories for both teams from a simulation
        Returns: (team1_experiences, team2_experiences)
        """
        team1_experiences = []
        team2_experiences = []
        
        rounds = simulation.get('Rounds', [])
        final_score = simulation.get('Score', [0, 0])
        team1_won = final_score[0] > final_score[1]
        
        # Track economic state (simplified - would need full economy simulation)
        team1_funds = 4000
        team2_funds = 4000
        team1_consec_losses = 0
        team2_consec_losses = 0
        
        for i, round_data in enumerate(rounds):
            round_num = round_data.get('RoundNumber', i + 1)
            team1_is_ct = round_data.get('is_t1_ct', True)
            
            # Calculate current scores
            team1_score = sum(1 for r in rounds[:i] if r.get('is_t1_winner_team', False))
            team2_score = sum(1 for r in rounds[:i] if not r.get('is_t1_winner_team', True))
            
            # Get previous round info
            if i > 0:
                prev_round = rounds[i - 1]
                prev_outcome = prev_round.get('Calc_Outcome', {})
                
                if prev_round.get('is_t1_winner_team', False):
                    team1_prev_survs = prev_outcome.get('CTSurvivors' if prev_round.get('is_t1_ct') else 'TSurvivors', 3)
                    team2_prev_survs = prev_outcome.get('TSurvivors' if prev_round.get('is_t1_ct') else 'CTSurvivors', 1)
                else:
                    team1_prev_survs = prev_outcome.get('TSurvivors' if prev_round.get('is_t1_ct') else 'CTSurvivors', 1)
                    team2_prev_survs = prev_outcome.get('CTSurvivors' if prev_round.get('is_t1_ct') else 'TSurvivors', 3)
                
                last_reason = prev_outcome.get('ReasonCode', 0)
                last_bomb = prev_outcome.get('BombPlanted', False)
            else:
                team1_prev_survs = 5
                team2_prev_survs = 5
                last_reason = 0
                last_bomb = False
            
            # Create states
            state1 = GameState(
                own_funds=team1_funds,
                own_score=team1_score,
                opponent_score=team2_score,
                own_survivors=team1_prev_survs,
                opponent_survivors=team2_prev_survs,
                consecutive_losses=team1_consec_losses,
                is_ct_side=team1_is_ct,
                round_number=round_num,
                half_length=15,
                last_round_reason=last_reason,
                last_bomb_planted=last_bomb
            )
            
            state2 = GameState(
                own_funds=team2_funds,
                own_score=team2_score,
                opponent_score=team1_score,
                own_survivors=team2_prev_survs,
                opponent_survivors=team1_prev_survs,
                consecutive_losses=team2_consec_losses,
                is_ct_side=not team1_is_ct,
                round_number=round_num,
                half_length=15,
                last_round_reason=last_reason,
                last_bomb_planted=last_bomb
            )
            
            # Extract actual investments from simulation
            outcome = round_data.get('Calc_Outcome', {})
            
            # Calculate equipment investment ratios (approximate from simulation data)
            ct_equip_total = sum(outcome.get('CTEquipmentPerPlayer', [0]))
            t_equip_total = sum(outcome.get('TEquipmentPerPlayer', [0]))
            
            # Approximate investment ratio (assuming they had some funds)
            # In reality, we'd need to track the exact economy
            if team1_is_ct:
                team1_equip = ct_equip_total
                team2_equip = t_equip_total
            else:
                team1_equip = t_equip_total
                team2_equip = ct_equip_total
            
            # Estimate action as ratio (capped at 1.0)
            action1 = min(1.0, team1_equip / max(team1_funds, 1))
            action2 = min(1.0, team2_equip / max(team2_funds, 1))
            
            # Round outcome
            team1_won_round = round_data.get('is_t1_winner_team', False)
            
            # Calculate round reward
            round_reward1 = 1.0 if team1_won_round else -1.0
            round_reward2 = -round_reward1
            
            # Create next states (will be filled in next iteration)
            next_state1 = None
            next_state2 = None
            
            if i < len(rounds) - 1:
                # Not the last round, next state will be created in next iteration
                team1_experiences.append((state1, action1, round_reward1))
                team2_experiences.append((state2, action2, round_reward2))
            else:
                # Last round, add match outcome reward
                match_reward1 = 10.0 if team1_won else -10.0
                match_reward2 = -match_reward1
                team1_experiences.append((state1, action1, round_reward1 + match_reward1))
                team2_experiences.append((state2, action2, round_reward2 + match_reward2))
            
            # Update economy (simplified CS:GO economy rules)
            loss_bonus = [1400, 1900, 2400, 2900, 3400]
            
            if team1_won_round:
                team1_funds += 3250  # Win reward
                team1_funds -= team1_equip
                team1_consec_losses = 0
                team2_consec_losses = min(4, team2_consec_losses + 1)
                team2_funds += loss_bonus[team2_consec_losses]
                team2_funds -= team2_equip
            else:
                team2_funds += 3250
                team2_funds -= team2_equip
                team2_consec_losses = 0
                team1_consec_losses = min(4, team1_consec_losses + 1)
                team1_funds += loss_bonus[team1_consec_losses]
                team1_funds -= team1_equip
            
            # Add passive income and cap
            team1_funds = max(0, min(50000, team1_funds + 1400))
            team2_funds = max(0, min(50000, team2_funds + 1400))
        
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
        Collect experiences from simulation files
        """
        sim_files = self.data_loader.sim_files
        if n_simulations:
            sim_files = sim_files[:n_simulations]
        
        print(f"\n📥 Collecting experiences from {len(sim_files)} simulations...")
        
        total_rounds = 0
        winning_experiences = []
        losing_experiences = []
        
        for sim_file in tqdm(sim_files, desc="Loading simulations"):
            try:
                sim_data = self.data_loader.load_simulation(sim_file)
                team1_exp, team2_exp = self.data_loader.extract_trajectories(sim_data)
                
                # Separate winning and losing team experiences
                final_score = sim_data.get('Score', [0, 0])
                if final_score[0] > final_score[1]:
                    winning_experiences.extend(team1_exp)
                    losing_experiences.extend(team2_exp)
                else:
                    winning_experiences.extend(team2_exp)
                    losing_experiences.extend(team1_exp)
                
                total_rounds += len(team1_exp)
                
            except Exception as e:
                print(f"Error processing {sim_file.name}: {e}")
                continue
        
        print(f"\n✓ Collected {total_rounds} rounds of experience")
        print(f"  Winning team experiences: {len(winning_experiences)}")
        print(f"  Losing team experiences: {len(losing_experiences)}")
        
        return winning_experiences, losing_experiences
    
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
        
        print(f"\n🎯 Training {strategy.name}...")
        print(f"  Total experiences: {len(experiences)}")
        print(f"  Epochs: {n_epochs}")
        
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
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CS:GO Economic ABM - Training from Simulation Data")
    print("=" * 70)
    
    # Load ABM data
    data_loader = ABMDataLoader(args.results_dir)
    
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
    print(f"\n🤖 Initializing strategies: {', '.join(args.strategies)}")
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
    
    # Train each strategy
    for name, strategy in strategies.items():
        try:
            trainer.train_strategy(
                strategy,
                training_experiences,
                n_epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            # Save model
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"{name}_abm_trained.pt")
            strategy.save(save_path)
            print(f"  ✓ Saved to {save_path}")
            
        except Exception as e:
            print(f"  ✗ Error training {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Models saved to: {args.save_dir}")
    print("=" * 70)
    
    # Save training info
    info = {
        "results_dir": args.results_dir,
        "n_simulations": len(data_loader.sim_files) if not args.n_simulations else args.n_simulations,
        "total_experiences": len(training_experiences),
        "winning_experiences": len(winning_exp),
        "losing_experiences": len(losing_exp),
        "use_winning_only": args.use_winning_only,
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
