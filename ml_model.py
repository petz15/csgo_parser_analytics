"""
CS:GO Economic ABM - Reinforcement Learning Model

This module implements various RL strategies for optimal economic decision-making
in a simplified CS:GO environment where teams decide how much to invest in equipment.

Key Features:
- State: team funds, scores, survivors, consecutive losses, side, round info
- Action: continuous investment amount (0.0 to 1.0 of available funds)
- Reward: match win/loss outcome
- Methods: DQN, PPO, Policy Gradient, and baseline strategies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from collections import deque, namedtuple
import json
import pickle
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
import xgboost as xgb
import joblib


# ============================================================================
# Configuration and Data Structures
# ============================================================================

@dataclass
class GameState:
    """Represents the observable game state for a team (legal/observable information only)"""
    own_funds: float
    own_score: int
    opponent_score: int
    own_survivors: int
    opponent_survivors: int
    consecutive_losses: int
    is_ct_side: bool
    round_number: int
    half_length: int
    last_round_reason: int  # 0=first round, 1-4=reason codes
    last_bomb_planted: bool
    own_starting_equipment: float = 0.0  # Equipment value at round start
    
    @property
    def score_diff(self) -> int:
        """Calculated: own_score - opponent_score"""
        return self.own_score - self.opponent_score
    
    def to_array(self) -> np.ndarray:
        """Convert state to normalized feature vector (13 features)"""
        return np.array([
            self.own_funds / 999999.0,  # Normalize to [0, 1]
            self.own_score / 16.0,
            self.opponent_score / 16.0,
            self.own_survivors / 5.0,
            self.opponent_survivors / 5.0,
            min(self.consecutive_losses, 5) / 5.0,
            1.0 if self.is_ct_side else 0.0,
            self.round_number / 30.0,  # Max ~30 rounds with OT
            self.half_length / 15.0,
            self.last_round_reason / 4.0,
            1.0 if self.last_bomb_planted else 0.0,
            self.own_starting_equipment / 27000.0,  # Max ~27k for full team buy
            (self.score_diff + 15.0) / 30.0,  # Normalize to [0, 1] from [-15, +15]
        ], dtype=np.float32)
    
    @property
    def state_dim(self) -> int:
        return 13


@dataclass
class ForbiddenGameState(GameState):
    """
    Extended game state with 'forbidden' information (opponent's private data).
    
    This should NOT be used in real gameplay or fair AI training, but can be useful for:
    - Research and analysis
    - Understanding theoretical optimal strategies
    - Creating benchmark models
    - Analyzing information asymmetry
    """
    opponent_funds: float = 0.0  # FORBIDDEN: Opponent's money (not visible in-game)
    opponent_starting_equipment: float = 0.0  # FORBIDDEN: Opponent's equipment value
    
    def to_array(self) -> np.ndarray:
        """Convert state to normalized feature vector (15 features including forbidden info)"""
        # Start with base features
        base_array = np.array([
            self.own_funds / 999999.0,
            self.own_score / 16.0,
            self.opponent_score / 16.0,
            self.own_survivors / 5.0,
            self.opponent_survivors / 5.0,
            min(self.consecutive_losses, 5) / 5.0,
            1.0 if self.is_ct_side else 0.0,
            self.round_number / 30.0,
            self.half_length / 15.0,
            self.last_round_reason / 4.0,
            1.0 if self.last_bomb_planted else 0.0,
            self.own_starting_equipment / 999999.0,
            (self.score_diff + 15.0) / 30.0,
            # FORBIDDEN features below
            self.opponent_funds / 999999.0,
            self.opponent_starting_equipment / 999999.0,
        ], dtype=np.float32)
        return base_array
    
    @property
    def state_dim(self) -> int:
        return 15


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


# ============================================================================
# Baseline Strategies (No Learning)
# ============================================================================

class BaseStrategy:
    """Base class for all strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.match_history = []
    
    def select_action(self, state: GameState, training: bool = True) -> float:
        """Return investment ratio (0.0 to 1.0)"""
        raise NotImplementedError
    
    def update(self, experience: Experience):
        """Update strategy based on experience (no-op for baselines)"""
        pass
    
    def record_match_result(self, won: bool):
        """Record match outcome"""
        self.match_history.append(1 if won else 0)
    
    def get_win_rate(self) -> float:
        """Calculate win rate"""
        if not self.match_history:
            return 0.0
        return sum(self.match_history) / len(self.match_history)
    
    def save(self, path: str):
        """Save strategy state"""
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load(self, path: str):
        """Load strategy state"""
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))


class FullBuyStrategy(BaseStrategy):
    """Always invest all available funds"""
    
    def __init__(self):
        super().__init__("FullBuy")
    
    def select_action(self, state: GameState, training: bool = True) -> float:
        return 1.0  # 100% investment


class ConservativeStrategy(BaseStrategy):
    """Invest 50% of available funds"""
    
    def __init__(self):
        super().__init__("Conservative")
    
    def select_action(self, state: GameState, training: bool = True) -> float:
        return 0.5  # 50% investment


class RandomStrategy(BaseStrategy):
    """Random investment between 0% and 100%"""
    
    def __init__(self):
        super().__init__("Random")
    
    def select_action(self, state: GameState, training: bool = True) -> float:
        return np.random.uniform(0.0, 1.0)


class AdaptiveThresholdStrategy(BaseStrategy):
    """
    Adaptive strategy based on game situation:
    - High investment when winning or tied
    - Low investment after losses to preserve economy
    - Considers consecutive losses for bonus recovery
    """
    
    def __init__(self):
        super().__init__("AdaptiveThreshold")
    
    def select_action(self, state: GameState, training: bool = True) -> float:
        score_diff = state.own_score - state.opponent_score
        
        # If winning significantly, maintain pressure
        if score_diff >= 3:
            return 0.9
        
        # If losing significantly and low funds, save
        if score_diff <= -3 and state.own_funds < 10000:
            return 0.2
        
        # After multiple losses, might want to save for bonus
        if state.consecutive_losses >= 4:
            if state.own_funds < 15000:
                return 0.3  # Light investment, wait for bonus
            else:
                return 0.8  # Have enough for good buy
        
        # Close game, balanced investment
        if abs(score_diff) <= 2:
            if state.own_funds > 20000:
                return 0.7
            else:
                return 0.5
        
        # Default moderate investment
        return 0.6


class MomentumStrategy(BaseStrategy):
    """
    Investment based on recent momentum (survivors and round wins)
    """
    
    def __init__(self):
        super().__init__("Momentum")
    
    def select_action(self, state: GameState, training: bool = True) -> float:
        # Base investment on survivor count (proxy for round performance)
        survivor_ratio = state.own_survivors / 5.0
        
        # Adjust based on funds availability
        funds_ratio = state.own_funds / 30000.0  # Target reference
        
        # Calculate momentum score
        momentum = (survivor_ratio + funds_ratio) / 2.0
        
        # Invest proportional to momentum, clamped
        investment = 0.3 + momentum * 0.6
        return np.clip(investment, 0.0, 1.0)


# ============================================================================
# Deep Q-Network (DQN) Implementation
# ============================================================================

class QNetwork(nn.Module):
    """
    Q-Network for approximating Q(s, a) values
    Uses discretized action space for DQN
    """
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dims: List[int] = [128, 128, 64]):
        super(QNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_actions))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state)


class DQNStrategy(BaseStrategy):
    """
    Deep Q-Network strategy with discretized actions
    Actions: [0.0, 0.1, 0.2, ..., 0.9, 1.0] investment ratios
    """
    
    def __init__(self, state_dim: int = 13, n_actions: int = 11, 
                 lr: float = 0.0003, gamma: float = 0.99, 
                 epsilon_start: float = 1.0, epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.9995, batch_size: int = 64,
                 buffer_size: int = 50000, target_update_freq: int = 100):
        super().__init__("DQN")
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.action_values = np.linspace(0.0, 1.0, n_actions)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks
        self.q_network = QNetwork(state_dim, n_actions).to(self.device)
        self.target_network = QNetwork(state_dim, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Experience replay
        self.memory = deque(maxlen=buffer_size)
        self.update_count = 0
        
        # Training stats
        self.losses = []
        
    def select_action(self, state: GameState, training: bool = True) -> float:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action according to Q-network
            state_tensor = torch.FloatTensor(state.to_array()).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        return self.action_values[action_idx]
    
    def update(self, experience: Experience):
        """Store experience and train if enough samples"""
        self.memory.append(experience)
        
        if len(self.memory) >= self.batch_size:
            self._train_step()
    
    def _train_step(self):
        """Perform one training step"""
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([e.state.to_array() for e in batch])).to(self.device)
        actions = torch.LongTensor([np.argmin(np.abs(self.action_values - e.action)) for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state.to_array() if e.next_state else np.zeros(self.state_dim) for e in batch])).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        self.update_count += 1
        
        # Update target network
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'match_history': self.match_history,
            'losses': self.losses,
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.match_history = checkpoint['match_history']
        self.losses = checkpoint['losses']


# ============================================================================
# Policy Gradient Implementation (Continuous Actions)
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Policy network outputting Beta distribution parameters for continuous actions
    Beta distribution is ideal for bounded continuous actions [0, 1]
    """
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 128, 64]):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Beta distribution parameters (alpha, beta)
        self.alpha_head = nn.Linear(prev_dim, 1)
        self.beta_head = nn.Linear(prev_dim, 1)
        
    def forward(self, state):
        features = self.shared_layers(state)
        # Softplus ensures positive parameters for Beta distribution
        alpha = F.softplus(self.alpha_head(features)) + 1.0
        beta = F.softplus(self.beta_head(features)) + 1.0
        return alpha, beta


class ValueNetwork(nn.Module):
    """Value network for advantage estimation"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 128, 64]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state).squeeze(-1)


class PPOStrategy(BaseStrategy):
    """
    Proximal Policy Optimization with continuous actions
    Uses Beta distribution for bounded continuous action space
    """
    
    def __init__(self, state_dim: int = 13, lr: float = 0.0003,
                 gamma: float = 0.99, lambda_gae: float = 0.95,
                 epsilon_clip: float = 0.2, epochs: int = 10,
                 batch_size: int = 64, buffer_size: int = 2048):
        super().__init__("PPO")
        
        self.state_dim = state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy = PolicyNetwork(state_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.epsilon_clip = epsilon_clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Training stats
        self.policy_losses = []
        self.value_losses = []
        
    def select_action(self, state: GameState, training: bool = True) -> float:
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state.to_array()).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            alpha, beta = self.policy(state_tensor)
            value = self.value(state_tensor)
        
        # Sample from Beta distribution
        dist = Beta(alpha, beta)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if training:
            # Store for training
            self.states.append(state.to_array())
            self.actions.append(action.item())
            self.values.append(value.item())
            self.log_probs.append(log_prob.item())
        
        return action.item()
    
    def store_reward(self, reward: float, done: bool):
        """Store reward after environment step"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self, experience: Experience = None):
        """Train on collected trajectory"""
        if len(self.states) < self.batch_size:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Compute returns and advantages using GAE
        returns, advantages = self._compute_gae()
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.epochs):
            # Random mini-batches
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Compute current policy log probs
                alpha, beta = self.policy(batch_states)
                dist = Beta(alpha, beta)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Ratio for PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                
                # Value loss
                values = self.value(batch_states)
                value_loss = F.mse_loss(values, batch_returns)
                
                # Optimize policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()
                
                # Optimize value
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.value_optimizer.step()
                
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return np.array(returns), np.array(advantages)
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'match_history': self.match_history,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.match_history = checkpoint['match_history']
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']


# ============================================================================
# Simple Policy Gradient (REINFORCE)
# ============================================================================

class REINFORCEStrategy(BaseStrategy):
    """
    Simple policy gradient (REINFORCE) with continuous actions
    Simpler than PPO but can be less stable
    """
    
    def __init__(self, state_dim: int = 13, lr: float = 0.001, gamma: float = 0.99):
        super().__init__("REINFORCE")
        
        self.state_dim = state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy network
        self.policy = PolicyNetwork(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        
        # Episode buffer
        self.states = []
        self.actions = []
        self.rewards = []
        
        # Training stats
        self.losses = []
    
    def select_action(self, state: GameState, training: bool = True) -> float:
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state.to_array()).unsqueeze(0).to(self.device)
        
        alpha, beta = self.policy(state_tensor)
        dist = Beta(alpha, beta)
        action = dist.sample()
        
        if training:
            self.states.append(state_tensor)
            self.actions.append(action)
        
        return action.item()
    
    def store_reward(self, reward: float):
        """Store reward"""
        self.rewards.append(reward)
    
    def update(self, experience: Experience = None):
        """Update policy at end of episode"""
        if len(self.states) == 0:
            return
        
        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy gradient
        loss = 0
        for state, action, G in zip(self.states, self.actions, returns):
            alpha, beta = self.policy(state)
            dist = Beta(alpha, beta)
            log_prob = dist.log_prob(action)
            loss += -log_prob * G
        
        loss = loss / len(self.states)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'match_history': self.match_history,
            'losses': self.losses,
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.match_history = checkpoint['match_history']
        self.losses = checkpoint['losses']


# ============================================================================
# Incremental Learning Strategy (SGD-based)
# ============================================================================

class SGDStrategy(BaseStrategy):
    """
    Online learning strategy using Stochastic Gradient Descent
    Supports incremental updates with partial_fit for continuous improvement
    """
    
    def __init__(self, learning_rate: float = 0.01, name: str = "SGD"):
        super().__init__(name)
        self.learning_rate = learning_rate
        self.state_dim = 11
        
        # Use PyTorch for online learning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Simple feedforward network for regression (outputs investment ratio)
        self.model = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output in [0, 1]
        ).to(self.device)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.losses = []
        
        # Buffer for mini-batch updates
        self.buffer = []
        self.batch_size = 32
    
    def select_action(self, state: GameState, training: bool = True) -> float:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state.to_array()).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.model(state_tensor).item()
        
        # Add exploration noise during training
        if training:
            action = np.clip(action + np.random.normal(0, 0.1), 0, 1)
        
        return action
    
    def update(self, experience: Experience):
        """Incremental update with experience"""
        self.buffer.append(experience)
        
        # Update when buffer reaches batch size
        if len(self.buffer) >= self.batch_size:
            self._train_step()
            self.buffer = []
    
    def _train_step(self):
        """Train on buffered experiences"""
        if not self.buffer:
            return
        
        # Prepare batch
        states = torch.FloatTensor([exp.state.to_array() for exp in self.buffer]).to(self.device)
        actions = torch.FloatTensor([exp.action for exp in self.buffer]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in self.buffer]).unsqueeze(1).to(self.device)
        
        # Predict actions
        predicted_actions = self.model(states)
        
        # Loss: weighted MSE (reward-weighted behavioral cloning)
        # Positive rewards -> learn from these actions
        # Negative rewards -> learn to avoid these actions
        weights = torch.sigmoid(rewards)  # Convert rewards to weights [0, 1]
        loss = (weights * (predicted_actions - actions) ** 2).mean()
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'match_history': self.match_history,
            'losses': self.losses,
            'learning_rate': self.learning_rate,
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.match_history = checkpoint['match_history']
        self.losses = checkpoint['losses']
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)


# ============================================================================
# Tree-Based Strategy (Decision Tree)
# ============================================================================

class TreeStrategy(BaseStrategy):
    """
    Tree-based strategy using Decision Tree or Random Forest
    Good for interpretability and handling non-linear relationships
    """
    
    def __init__(self, use_forest: bool = True, name: str = None):
        if name is None:
            name = "RandomForest" if use_forest else "DecisionTree"
        super().__init__(name)
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.tree import DecisionTreeRegressor
            import joblib
            self.joblib = joblib
        except ImportError:
            raise ImportError("sklearn is required for TreeStrategy. Install with: pip install scikit-learn")
        
        self.use_forest = use_forest
        self.state_dim = 11
        
        # Create model
        if use_forest:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
        
        # Buffer for training data
        self.X_buffer = []
        self.y_buffer = []
        self.is_fitted = False
        # RandomForest is much slower, use larger interval
        self.retrain_interval = 10000 if use_forest else 1000
    
    def select_action(self, state: GameState, training: bool = True) -> float:
        """Select action using tree model"""
        state_array = state.to_array().reshape(1, -1)
        
        if not self.is_fitted:
            # Before first training, use conservative strategy
            return 0.5
        
        try:
            action = self.model.predict(state_array)[0]
            action = np.clip(action, 0, 1)
            
            # Add small exploration noise during training
            if training:
                action = np.clip(action + np.random.normal(0, 0.05), 0, 1)
            
            return action
        except:
            return 0.5
    
    def update(self, experience: Experience):
        """Collect experience and retrain periodically"""
        # Add to buffer
        self.X_buffer.append(experience.state.to_array())
        self.y_buffer.append(experience.action)
        
        # Retrain when buffer is large enough
        if len(self.X_buffer) >= self.retrain_interval:
            self._retrain()
    
    def _retrain(self):
        """Retrain the tree model with all collected data"""
        if len(self.X_buffer) < 10:
            return
        
        X = np.array(self.X_buffer)
        y = np.array(self.y_buffer)
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        
        print(f"  [{self.name}] Retrained on {len(X)} samples")
        
        # Clear buffers to prevent continuous retraining
        self.X_buffer = []
        self.y_buffer = []
    
    def finalize_training(self):
        """Final training with all accumulated data"""
        print(f"  [{self.name}] Finalizing: X_buffer size = {len(self.X_buffer)}, y_buffer size = {len(self.y_buffer)}")
        if self.X_buffer:
            self._retrain()
        else:
            print(f"  [{self.name}] WARNING: No data in buffer to train on!")
    
    def save(self, path: str):
        """Save model"""
        import joblib
        if self.is_fitted:
            joblib.dump({
                'model': self.model,
                'match_history': self.match_history,
                'is_fitted': self.is_fitted,
                'use_forest': self.use_forest,
            }, path)
        else:
            # Save untrained model state (still use joblib for consistency)
            joblib.dump({
                'match_history': self.match_history,
                'is_fitted': False,
                'use_forest': self.use_forest,
                'model': None,
            }, path)
    
    def load(self, path: str):
        """Load model"""
        import joblib
        try:
            data = joblib.load(path)
            if data.get('model') is not None:
                self.model = data['model']
            self.match_history = data.get('match_history', [])
            self.is_fitted = data.get('is_fitted', False)
            self.use_forest = data.get('use_forest', self.use_forest)
        except Exception as e:
            # Fallback to pickle for legacy files
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
                if data.get('model') is not None:
                    self.model = data['model']
                self.match_history = data.get('match_history', [])
                self.is_fitted = data.get('is_fitted', False)
                self.use_forest = data.get('use_forest', self.use_forest)


# ============================================================================
# Logistic Regression Strategy
# ============================================================================

class LogisticStrategy(BaseStrategy):
    """
    Logistic Regression strategy using linear model with sigmoid activation
    Simple interpretable model that learns linear combinations of features
    Output is bounded to [0, 1] through sigmoid transformation
    """
    
    def __init__(self, name: str = "Logistic"):
        super().__init__(name)
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=1.0, random_state=42)
        self.is_fitted = False
        self.X_buffer = []
        self.y_buffer = []
        self.retrain_interval = 1000
    
    def select_action(self, state: GameState, training: bool = True) -> float:
        """Select action using logistic regression"""
        if not self.is_fitted:
            # Before training, use moderate investment
            return 0.65
        
        # Get state features and predict
        state_array = state.to_array().reshape(1, -1)
        
        # Predict raw value and apply sigmoid to bound to [0, 1]
        raw_pred = self.model.predict(state_array)[0]
        investment = self._sigmoid(raw_pred)
        
        return float(np.clip(investment, 0.0, 1.0))
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function to map R -> [0, 1]"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
    
    def _inverse_sigmoid(self, y: float) -> float:
        """Inverse sigmoid (logit) to map [0, 1] -> R"""
        # Clip y to avoid log(0)
        y = np.clip(y, 1e-7, 1 - 1e-7)
        return np.log(y / (1 - y))
    
    def update(self, experience: Experience):
        """Add experience to buffer and retrain periodically"""
        state_array = experience.state.to_array()
        # Transform action through inverse sigmoid for training
        target = self._inverse_sigmoid(experience.action)
        
        self.X_buffer.append(state_array)
        self.y_buffer.append(target)
        
        # Retrain periodically
        if len(self.X_buffer) >= self.retrain_interval:
            self._retrain()
    
    def _retrain(self):
        """Retrain model on buffered experiences"""
        if len(self.X_buffer) < 10:
            return
        
        X = np.array(self.X_buffer)
        y = np.array(self.y_buffer)
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Clear buffers
        self.X_buffer = []
        self.y_buffer = []
    
    def finalize_training(self):
        """Train on all remaining buffered data"""
        if len(self.X_buffer) > 0:
            self._retrain()
    
    def save(self, path: str):
        """Save model to disk"""
        # Change extension to .joblib if it's .pt
        if path.endswith('.pt'):
            path = path.replace('.pt', '.joblib')
        
        data = {
            'model': self.model if self.is_fitted else None,
            'is_fitted': self.is_fitted,
            'match_history': self.match_history,
            'retrain_interval': self.retrain_interval
        }
        joblib.dump(data, path)
        print(f"Saved {self.name} model to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        data = joblib.load(path)
        if isinstance(data, dict):
            if data.get('model') is not None:
                self.model = data['model']
            self.is_fitted = data.get('is_fitted', False)
            self.match_history = data.get('match_history', [])
            self.retrain_interval = data.get('retrain_interval', 1000)
        return data


# ============================================================================
# XGBoost Strategy
# ============================================================================

class XGBoostStrategy(BaseStrategy):
    """
    XGBoost-based strategy using gradient boosting for regression
    Offers strong performance and handles feature interactions well
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, name: str = "XGBoost"):
        super().__init__(name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        
        # XGBoost regressor
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        self.is_fitted = False
        self.X_buffer = []  # Store state arrays
        self.y_buffer = []  # Store actions
        self.retrain_interval = 1000  # Retrain every N experiences
    
    def select_action(self, state: GameState, training: bool = True) -> float:
        """Select action using XGBoost model"""
        if not self.is_fitted:
            # Default to adaptive strategy before training
            return 0.7 if state.own_score >= state.opponent_score else 0.5
        
        # Convert state to feature vector
        state_array = state.to_array().reshape(1, -1)
        
        # Predict action
        action = self.model.predict(state_array)[0]
        return np.clip(action, 0.0, 1.0)
    
    def update(self, experience: Experience):
        """Store experience and retrain periodically"""
        self.X_buffer.append(experience.state.to_array())
        self.y_buffer.append(experience.action)
        
        # Retrain periodically
        if len(self.X_buffer) >= self.retrain_interval:
            self._retrain()
    
    def _retrain(self):
        """Retrain XGBoost model on collected experiences"""
        if len(self.X_buffer) < 10:
            return
        
        # Prepare training data
        X = np.array(self.X_buffer)
        y = np.array(self.y_buffer)
        
        # Train model
        self.model.fit(X, y)
        self.is_fitted = True
        
        print(f"  [{self.name}] Retrained on {len(X)} samples")
        
        # Clear buffers to prevent memory growth
        self.X_buffer = []
        self.y_buffer = []
    
    def finalize_training(self):
        """Final training on all buffered experiences"""
        print(f"  [{self.name}] Finalizing: X_buffer size = {len(self.X_buffer)}, y_buffer size = {len(self.y_buffer)}")
        if self.X_buffer:
            self._retrain()
        else:
            print(f"  [{self.name}] No data to finalize training")
    
    def save(self, path: str):
        """Save XGBoost model"""
        data = {
            'model': self.model,
            'is_fitted': self.is_fitted,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'match_history': self.match_history
        }
        joblib.dump(data, path)
    
    def load(self, path: str):
        """Load XGBoost model"""
        data = joblib.load(path)
        self.model = data['model']
        self.is_fitted = data['is_fitted']
        self.n_estimators = data.get('n_estimators', 100)
        self.max_depth = data.get('max_depth', 6)
        self.learning_rate = data.get('learning_rate', 0.1)
        self.match_history = data.get('match_history', [])
        return data


# ============================================================================
# Strategy Factory and Utilities
# ============================================================================

class StrategyFactory:
    """Factory for creating strategies"""
    
    @staticmethod
    def create(strategy_type: str, **kwargs) -> BaseStrategy:
        """Create strategy by type"""
        strategies = {
            'full_buy': FullBuyStrategy,
            'conservative': ConservativeStrategy,
            'random': RandomStrategy,
            'adaptive': AdaptiveThresholdStrategy,
            'momentum': MomentumStrategy,
            'dqn': DQNStrategy,
            'ppo': PPOStrategy,
            'reinforce': REINFORCEStrategy,
            'sgd': SGDStrategy,
            'tree': lambda: TreeStrategy(use_forest=False),
            'forest': lambda: TreeStrategy(use_forest=True),
            'logistic': LogisticStrategy,
            'xgboost': XGBoostStrategy,
        }
        
        if strategy_type.lower() not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_type}")
        
        return strategies[strategy_type.lower()](**kwargs)


def save_training_results(strategies: Dict[str, BaseStrategy], filepath: str):
    """Save training results to JSON"""
    results = {
        'strategies': {},
        'summary': {}
    }
    
    for name, strategy in strategies.items():
        win_rate = strategy.get_win_rate()
        results['strategies'][name] = {
            'name': strategy.name,
            'matches_played': len(strategy.match_history),
            'wins': sum(strategy.match_history),
            'losses': len(strategy.match_history) - sum(strategy.match_history),
            'win_rate': win_rate,
        }
        
        # Add losses if available
        if hasattr(strategy, 'losses') and strategy.losses:
            results['strategies'][name]['avg_loss'] = np.mean(strategy.losses[-1000:])
            results['strategies'][name]['total_updates'] = len(strategy.losses)
    
    # Summary statistics
    results['summary']['best_strategy'] = max(
        results['strategies'].items(), 
        key=lambda x: x[1]['win_rate']
    )[0]
    results['summary']['worst_strategy'] = min(
        results['strategies'].items(), 
        key=lambda x: x[1]['win_rate']
    )[0]
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining results saved to {filepath}")
    print(f"Best strategy: {results['summary']['best_strategy']} "
          f"(WR: {results['strategies'][results['summary']['best_strategy']]['win_rate']:.2%})")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("CS:GO Economic ABM - Reinforcement Learning Strategies")
    print("=" * 70)
    
    # Create example state (regular observable state)
    state = GameState(
        own_funds=10000,
        own_score=7,
        opponent_score=5,
        own_survivors=3,
        opponent_survivors=2,
        consecutive_losses=1,
        is_ct_side=True,
        round_number=12,
        half_length=15,
        last_round_reason=4,
        last_bomb_planted=False,
        own_starting_equipment=18500
    )
    
    print(f"\n=== Regular GameState (Observable Information) ===")
    print(f"  Own Funds: ${state.own_funds}")
    print(f"  Score: {state.own_score}-{state.opponent_score} (diff: {state.score_diff:+d})")
    print(f"  Survivors: {state.own_survivors} vs {state.opponent_survivors}")
    print(f"  Consecutive Losses: {state.consecutive_losses}")
    print(f"  Side: {'CT' if state.is_ct_side else 'T'}")
    print(f"  Round: {state.round_number}/{state.half_length * 2}")
    print(f"  Own Equipment: ${state.own_starting_equipment}")
    print(f"  State Dimension: {state.state_dim} features")
    
    # Create example forbidden state (includes opponent's private info)
    forbidden_state = ForbiddenGameState(
        own_funds=10000,
        own_score=7,
        opponent_score=5,
        own_survivors=3,
        opponent_survivors=2,
        consecutive_losses=1,
        is_ct_side=True,
        round_number=12,
        half_length=15,
        last_round_reason=4,
        last_bomb_planted=False,
        own_starting_equipment=18500,
        opponent_funds=8000,  # FORBIDDEN
        opponent_starting_equipment=12000  # FORBIDDEN
    )
    
    print(f"\n=== ForbiddenGameState (Includes Opponent's Private Data) ===")
    print(f"  ⚠️  WARNING: This state includes 'forbidden' information!")
    print(f"  Own Funds: ${forbidden_state.own_funds}")
    print(f"  Opponent Funds: ${forbidden_state.opponent_funds} [FORBIDDEN]")
    print(f"  Own Equipment: ${forbidden_state.own_starting_equipment}")
    print(f"  Opponent Equipment: ${forbidden_state.opponent_starting_equipment} [FORBIDDEN]")
    print(f"  State Dimension: {forbidden_state.state_dim} features")
    
    print(f"\nInitializing strategies...")
    
    # Create different strategies
    strategies = {
        'FullBuy': FullBuyStrategy(),
        'Conservative': ConservativeStrategy(),
        'Random': RandomStrategy(),
        'Adaptive': AdaptiveThresholdStrategy(),
        'Momentum': MomentumStrategy(),
        'DQN': DQNStrategy(),
        'PPO': PPOStrategy(),
        'REINFORCE': REINFORCEStrategy(),
        'SGD': SGDStrategy(),
        'Tree': TreeStrategy(use_forest=False),
        'Forest': TreeStrategy(use_forest=True),
        'Logistic': LogisticStrategy(),
        'XGBoost': XGBoostStrategy(),
    }
    
    print(f"\nStrategy Actions for Example State:")
    print("-" * 70)
    
    for name, strategy in strategies.items():
        action = strategy.select_action(state, training=False)
        investment = state.own_funds * action
        print(f"{name:15s}: Invest {action:5.1%} (${investment:,.0f})")
    
    print("\n" + "=" * 70)
    print("Ready for training! Use trainer.py to train these strategies.")
    print("=" * 70)
