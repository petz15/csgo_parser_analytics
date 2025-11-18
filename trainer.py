"""
CS:GO Economic ABM - Training Script

This script trains RL strategies using simulation data from the ABM.
It can train against fixed strategies or use self-play.
"""

import numpy as np
import json
import os
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from ml_model import (
    GameState, Experience, BaseStrategy, StrategyFactory,
    save_training_results
)


# ============================================================================
# ABM Simulation Interface
# ============================================================================

class ABMSimulationData:
    """
    Interface to load and analyze ABM simulation results
    """
    
    def __init__(self, results_folder: str):
        self.results_folder = results_folder
        self.summary = None
        self.simulations = []
        
        self._load_data()
    
    def _load_data(self):
        """Load simulation summary and individual simulations"""
        summary_path = os.path.join(self.results_folder, 'simulation_summary.json')
        
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        
        with open(summary_path, 'r') as f:
            self.summary = json.load(f)
        
        print(f"Loaded summary from {self.results_folder}")
        print(f"  Total simulations: {self.summary['total_simulations']}")
        print(f"  Team1 win rate: {self.summary['team1_win_rate']:.2f}%")
        print(f"  Team2 win rate: {self.summary['team2_win_rate']:.2f}%")
    
    def load_simulation(self, sim_id: int) -> Dict:
        """Load a specific simulation file"""
        # Find the file matching this simulation ID
        sim_files = [f for f in os.listdir(self.results_folder) 
                     if f.startswith(f'sim_{sim_id}_')]
        
        if not sim_files:
            raise FileNotFoundError(f"No simulation file found for ID {sim_id}")
        
        sim_path = os.path.join(self.results_folder, sim_files[0])
        with open(sim_path, 'r') as f:
            return json.load(f)
    
    def extract_round_states(self, simulation: Dict) -> List[Tuple[GameState, GameState, Dict]]:
        """
        Extract game states for both teams from each round
        Returns: List of (team1_state, team2_state, round_outcome)
        """
        states = []
        
        for i, round_data in enumerate(simulation['Rounds']):
            # Determine which team is CT/T
            team1_is_ct = round_data['is_t1_ct']
            team2_is_ct = not team1_is_ct
            
            # Calculate current score before this round
            team1_score = simulation['Score'][0] if i == len(simulation['Rounds']) - 1 else \
                         sum(1 for r in simulation['Rounds'][:i] if r['is_t1_winner_team'])
            team2_score = simulation['Score'][1] if i == len(simulation['Rounds']) - 1 else \
                         sum(1 for r in simulation['Rounds'][:i] if not r['is_t1_winner_team'])
            
            # Get survivor counts from previous round
            prev_round = simulation['Rounds'][i-1] if i > 0 else None
            
            if prev_round:
                if prev_round['is_t1_winner_team']:
                    team1_prev_survivors = prev_round['Calc_Outcome']['CTSurvivors'] if prev_round['is_t1_ct'] else prev_round['Calc_Outcome']['TSurvivors']
                    team2_prev_survivors = prev_round['Calc_Outcome']['TSurvivors'] if prev_round['is_t1_ct'] else prev_round['Calc_Outcome']['CTSurvivors']
                else:
                    team1_prev_survivors = prev_round['Calc_Outcome']['TSurvivors'] if prev_round['is_t1_ct'] else prev_round['Calc_Outcome']['CTSurvivors']
                    team2_prev_survivors = prev_round['Calc_Outcome']['CTSurvivors'] if prev_round['is_t1_ct'] else prev_round['Calc_Outcome']['TSurvivors']
                
                last_reason = prev_round['Calc_Outcome']['ReasonCode']
                last_bomb = prev_round['Calc_Outcome']['BombPlanted']
            else:
                team1_prev_survivors = 5
                team2_prev_survivors = 5
                last_reason = 0
                last_bomb = False
            
            # Calculate consecutive losses (simplified - would need full history)
            # For now, estimate based on score difference
            team1_consec = max(0, min(5, team2_score - team1_score)) if team2_score > team1_score else 0
            team2_consec = max(0, min(5, team1_score - team2_score)) if team1_score > team2_score else 0
            
            # Note: We don't have exact fund information before decisions
            # This would need to be calculated based on CS:GO economy rules
            # For now, use placeholder values
            team1_funds = 20000  # Placeholder
            team2_funds = 20000  # Placeholder
            
            team1_state = GameState(
                own_funds=team1_funds,
                own_score=team1_score,
                opponent_score=team2_score,
                own_survivors=team1_prev_survivors,
                opponent_survivors=team2_prev_survivors,
                consecutive_losses=team1_consec,
                is_ct_side=team1_is_ct,
                round_number=round_data['RoundNumber'],
                half_length=15,
                last_round_reason=last_reason,
                last_bomb_planted=last_bomb
            )
            
            team2_state = GameState(
                own_funds=team2_funds,
                own_score=team2_score,
                opponent_score=team1_score,
                own_survivors=team2_prev_survivors,
                opponent_survivors=team1_prev_survivors,
                consecutive_losses=team2_consec,
                is_ct_side=team2_is_ct,
                round_number=round_data['RoundNumber'],
                half_length=15,
                last_round_reason=last_reason,
                last_bomb_planted=last_bomb
            )
            
            states.append((team1_state, team2_state, round_data['Calc_Outcome']))
        
        return states


# ============================================================================
# Training Environment
# ============================================================================

class TrainingEnvironment:
    """
    Training environment that simulates matches between strategies
    Uses distributions.json for outcome probabilities
    """
    
    def __init__(self, distributions_path: str = 'distributions.json'):
        self.distributions_path = distributions_path
        self.distributions = None
        
        self._load_distributions()
    
    def _load_distributions(self):
        """Load outcome distributions"""
        if not os.path.exists(self.distributions_path):
            print(f"Warning: distributions.json not found at {self.distributions_path}")
            print("Will use simplified simulation without outcome distributions")
            return
        
        with open(self.distributions_path, 'r') as f:
            self.distributions = json.load(f)
        
        print(f"Loaded distributions from {self.distributions_path}")
    
    def simulate_match(self, strategy1: BaseStrategy, strategy2: BaseStrategy,
                       half_length: int = 15, max_rounds: int = 30,
                       verbose: bool = False) -> Tuple[bool, List[Experience], List[Experience]]:
        """
        Simulate a match between two strategies
        Returns: (strategy1_won, strategy1_experiences, strategy2_experiences)
        """
        
        # Initialize match state
        team1_score = 0
        team2_score = 0
        team1_funds = 4000  # Starting pistol round funds
        team2_funds = 4000
        team1_consec_losses = 0
        team2_consec_losses = 0
        
        team1_experiences = []
        team2_experiences = []
        
        # Track states for experience replay
        team1_prev_state = None
        team2_prev_state = None
        team1_prev_action = None
        team2_prev_action = None
        
        for round_num in range(1, max_rounds + 1):
            # Check if match is over
            if team1_score >= 16 or team2_score >= 16:
                break
            
            # Determine sides (swap at half)
            team1_is_ct = round_num <= half_length
            
            # Previous round info
            if round_num == 1:
                last_reason = 0
                last_bomb = False
                team1_survivors = 5
                team2_survivors = 5
            else:
                # Use previous outcome
                last_reason = prev_reason
                last_bomb = prev_bomb
                team1_survivors = prev_team1_survivors
                team2_survivors = prev_team2_survivors
            
            # Create states
            state1 = GameState(
                own_funds=team1_funds,
                own_score=team1_score,
                opponent_score=team2_score,
                own_survivors=team1_survivors,
                opponent_survivors=team2_survivors,
                consecutive_losses=team1_consec_losses,
                is_ct_side=team1_is_ct,
                round_number=round_num,
                half_length=half_length,
                last_round_reason=last_reason,
                last_bomb_planted=last_bomb
            )
            
            state2 = GameState(
                own_funds=team2_funds,
                own_score=team2_score,
                opponent_score=team1_score,
                own_survivors=team2_survivors,
                opponent_survivors=team1_survivors,
                consecutive_losses=team2_consec_losses,
                is_ct_side=not team1_is_ct,
                round_number=round_num,
                half_length=half_length,
                last_round_reason=last_reason,
                last_bomb_planted=last_bomb
            )
            
            # Store previous experience
            if team1_prev_state is not None:
                exp1 = Experience(team1_prev_state, team1_prev_action, 0, state1, False)
                team1_experiences.append(exp1)
            
            if team2_prev_state is not None:
                exp2 = Experience(team2_prev_state, team2_prev_action, 0, state2, False)
                team2_experiences.append(exp2)
            
            # Get actions (investment ratios)
            action1 = strategy1.select_action(state1)
            action2 = strategy2.select_action(state2)
            
            # Calculate equipment invested
            team1_equipment = team1_funds * action1
            team2_equipment = team2_funds * action2
            
            # Simulate round outcome using CSF or simple probability
            team1_won_round = self._simulate_round_outcome(
                team1_equipment, team2_equipment, team1_is_ct
            )
            
            # Update scores
            if team1_won_round:
                team1_score += 1
                team1_consec_losses = 0
                team2_consec_losses = min(5, team2_consec_losses + 1)
            else:
                team2_score += 1
                team2_consec_losses = 0
                team1_consec_losses = min(5, team1_consec_losses + 1)
            
            # Calculate rewards (round outcome)
            round_reward_t1 = 1.0 if team1_won_round else -1.0
            round_reward_t2 = -round_reward_t1
            
            # Economy updates (simplified CS:GO economy)
            loss_bonus = [1400, 1900, 2400, 2900, 3400]
            
            if team1_won_round:
                team1_funds += 3250  # Win reward
                team1_funds -= team1_equipment
                prev_team1_survivors = np.random.randint(2, 6)  # Simplified
                prev_team2_survivors = np.random.randint(0, 3)
            else:
                team1_funds += loss_bonus[min(team1_consec_losses - 1, 4)] if team1_consec_losses > 0 else 1400
                team1_funds -= team1_equipment
                prev_team1_survivors = np.random.randint(0, 3)
                prev_team2_survivors = np.random.randint(2, 6)
            
            if team2_won_round:
                team2_funds += 3250
                team2_funds -= team2_equipment
            else:
                team2_funds += loss_bonus[min(team2_consec_losses - 1, 4)] if team2_consec_losses > 0 else 1400
                team2_funds -= team2_equipment
            
            # Cap funds
            team1_funds = max(0, min(50000, team1_funds + 1400))  # +1400 passive income
            team2_funds = max(0, min(50000, team2_funds + 1400))
            
            # Store for next iteration
            team1_prev_state = state1
            team2_prev_state = state2
            team1_prev_action = action1
            team2_prev_action = action2
            prev_reason = np.random.randint(1, 5)  # Simplified
            prev_bomb = np.random.random() < 0.3  # Simplified
            
            if verbose and round_num % 5 == 0:
                print(f"  Round {round_num}: {team1_score}-{team2_score} | "
                      f"Funds: ${team1_funds:.0f} vs ${team2_funds:.0f}")
        
        # Determine winner
        strategy1_won = team1_score > team2_score
        
        # Add final experiences with match outcome reward
        match_reward_t1 = 10.0 if strategy1_won else -10.0
        match_reward_t2 = -match_reward_t1
        
        # Update all experience rewards with final match reward
        for exp in team1_experiences:
            # Combine round reward with discounted match reward
            exp = Experience(exp.state, exp.action, match_reward_t1, exp.next_state, False)
        
        for exp in team2_experiences:
            exp = Experience(exp.state, exp.action, match_reward_t2, exp.next_state, False)
        
        # Mark last experience as done
        if team1_experiences:
            last_exp = team1_experiences[-1]
            team1_experiences[-1] = Experience(last_exp.state, last_exp.action, 
                                               match_reward_t1, last_exp.next_state, True)
        
        if team2_experiences:
            last_exp = team2_experiences[-1]
            team2_experiences[-1] = Experience(last_exp.state, last_exp.action,
                                               match_reward_t2, last_exp.next_state, True)
        
        return strategy1_won, team1_experiences, team2_experiences
    
    def _simulate_round_outcome(self, team1_equip: float, team2_equip: float, 
                                team1_is_ct: bool) -> bool:
        """
        Simulate round outcome based on equipment values
        Returns True if team1 wins
        """
        if self.distributions is None:
            # Simple probability based on equipment ratio
            r = 1.085  # From distributions metadata
            ct_equip = team1_equip if team1_is_ct else team2_equip
            t_equip = team2_equip if team1_is_ct else team1_equip
            
            if ct_equip + t_equip == 0:
                csf = 0.5
            else:
                csf = (ct_equip ** r) / ((ct_equip ** r) + (t_equip ** r))
            
            ct_wins = np.random.random() < csf
            team1_wins = ct_wins if team1_is_ct else not ct_wins
            return team1_wins
        
        # TODO: Use actual distributions from distributions.json
        # This would involve:
        # 1. Calculate CSF from equipment
        # 2. Look up outcome distribution for that CSF
        # 3. Sample from distribution
        
        # For now, use simplified version above
        r = self.distributions['metadata']['csf_r_value']
        ct_equip = team1_equip if team1_is_ct else team2_equip
        t_equip = team2_equip if team1_is_ct else team1_equip
        
        if ct_equip + t_equip == 0:
            csf = 0.5
        else:
            csf = (ct_equip ** r) / ((ct_equip ** r) + (t_equip ** r))
        
        ct_wins = np.random.random() < csf
        team1_wins = ct_wins if team1_is_ct else not ct_wins
        return team1_wins


# ============================================================================
# Training Loop
# ============================================================================

def train_strategies(strategies: Dict[str, BaseStrategy], 
                     env: TrainingEnvironment,
                     n_matches: int = 1000,
                     update_frequency: int = 1,
                     save_dir: str = 'models',
                     verbose: bool = True) -> Dict[str, BaseStrategy]:
    """
    Train strategies through self-play and vs baseline strategies
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    strategy_names = list(strategies.keys())
    n_strategies = len(strategy_names)
    
    print(f"\nTraining {n_strategies} strategies for {n_matches} matches...")
    print("=" * 70)
    
    # Training loop
    for match_idx in tqdm(range(n_matches), desc="Training"):
        # Randomly pair strategies
        if np.random.random() < 0.5 and n_strategies > 1:
            # Play against another strategy
            s1_name, s2_name = np.random.choice(strategy_names, 2, replace=False)
        else:
            # Self-play or vs random
            s1_name = np.random.choice(strategy_names)
            s2_name = np.random.choice(strategy_names)
        
        strategy1 = strategies[s1_name]
        strategy2 = strategies[s2_name]
        
        # Simulate match
        s1_won, s1_exp, s2_exp = env.simulate_match(
            strategy1, strategy2, verbose=False
        )
        
        # Record results
        strategy1.record_match_result(s1_won)
        strategy2.record_match_result(not s1_won)
        
        # Update strategies with experiences
        for exp in s1_exp:
            strategy1.update(exp)
        
        for exp in s2_exp:
            strategy2.update(exp)
        
        # Periodic updates for PPO/REINFORCE (episode-based)
        if (match_idx + 1) % update_frequency == 0:
            for strategy in strategies.values():
                if hasattr(strategy, 'store_reward'):
                    # These strategies need end-of-episode update
                    strategy.update()
        
        # Progress logging
        if verbose and (match_idx + 1) % 100 == 0:
            print(f"\nProgress at match {match_idx + 1}:")
            for name, strategy in strategies.items():
                wr = strategy.get_win_rate()
                matches = len(strategy.match_history)
                print(f"  {name:15s}: {wr:6.2%} WR ({matches} matches)")
    
    # Save models
    print("\nSaving trained models...")
    for name, strategy in strategies.items():
        model_path = os.path.join(save_dir, f"{name.lower()}_model.pt")
        strategy.save(model_path)
        print(f"  Saved {name} to {model_path}")
    
    return strategies


def evaluate_strategies(strategies: Dict[str, BaseStrategy],
                       env: TrainingEnvironment,
                       n_eval_matches: int = 100,
                       verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate strategies in head-to-head matches
    """
    
    print(f"\nEvaluating strategies over {n_eval_matches} matches...")
    print("=" * 70)
    
    results = {name: [] for name in strategies.keys()}
    
    for _ in tqdm(range(n_eval_matches), desc="Evaluating"):
        # Round-robin evaluation
        strategy_names = list(strategies.keys())
        for i, s1_name in enumerate(strategy_names):
            for s2_name in strategy_names[i+1:]:
                strategy1 = strategies[s1_name]
                strategy2 = strategies[s2_name]
                
                # Play match without training
                s1_won, _, _ = env.simulate_match(
                    strategy1, strategy2, verbose=False
                )
                
                results[s1_name].append(1 if s1_won else 0)
                results[s2_name].append(0 if s1_won else 1)
    
    # Calculate win rates
    win_rates = {name: np.mean(wins) if wins else 0.0 
                 for name, wins in results.items()}
    
    if verbose:
        print("\nEvaluation Results:")
        print("-" * 70)
        sorted_results = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
        for rank, (name, wr) in enumerate(sorted_results, 1):
            print(f"  {rank}. {name:15s}: {wr:6.2%} WR")
    
    return win_rates


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train CS:GO Economic ABM RL Strategies')
    parser.add_argument('--n-matches', type=int, default=1000,
                       help='Number of training matches')
    parser.add_argument('--eval-matches', type=int, default=100,
                       help='Number of evaluation matches')
    parser.add_argument('--strategies', nargs='+', 
                       default=['full_buy', 'adaptive', 'dqn', 'ppo'],
                       help='Strategies to train')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--results-dir', type=str, 
                       default='results_20251118_002441',
                       help='Directory with ABM simulation results')
    parser.add_argument('--no-train', action='store_true',
                       help='Skip training, only evaluate')
    
    args = parser.parse_args()
    
    print("CS:GO Economic ABM - Strategy Training")
    print("=" * 70)
    
    # Initialize environment
    env = TrainingEnvironment('distributions.json')
    
    # Create strategies
    print(f"\nInitializing strategies: {', '.join(args.strategies)}")
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
    
    # Training
    if not args.no_train:
        trained_strategies = train_strategies(
            strategies,
            env,
            n_matches=args.n_matches,
            save_dir=args.save_dir,
            verbose=True
        )
    else:
        print("\nSkipping training (--no-train)")
        trained_strategies = strategies
    
    # Evaluation
    eval_results = evaluate_strategies(
        trained_strategies,
        env,
        n_eval_matches=args.eval_matches,
        verbose=True
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"training_results_{timestamp}.json"
    save_training_results(trained_strategies, results_file)
    
    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
