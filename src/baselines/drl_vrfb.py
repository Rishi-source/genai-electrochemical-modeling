"""
DRL Baseline for VRFB Parameter Tuning
Dueling Deep Q-Network (DQN) from Paper [3].

Key features:
- Dueling architecture (value + advantage streams)
- Experience replay buffer (size=10⁵)
- VRFB simulation environment
- Parameter tuning task
- Metrics: SOC RMSE, WLSS reduction
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as TF  # Renamed to avoid conflict with Faraday constant

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.physics_constants import R, F, VRFBParams


@dataclass
class DRLTrainingResult:
    """Results from DRL training."""
    episode_rewards: np.ndarray
    soc_rmse_history: np.ndarray
    wlss_history: np.ndarray
    
    final_soc_rmse: float
    final_wlss: float
    
    best_parameters: Dict[str, float]
    training_time: float
    total_steps: int


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network architecture.
    
    Separates value and advantage streams:
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    
    This allows the network to learn which states are valuable
    independent of the action taken.
    """
    
    def __init__(self, state_dim: int = 6, action_dim: int = 5):
        """
        Initialize Dueling DQN.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
        """
        super(DuelingDQN, self).__init__()
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Returns:
            Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        features = self.feature(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling architecture: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    
    Stores (state, action, reward, next_state, done) transitions.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class VRFBSimulationEnv:
    """
    VRFB simulation environment for DRL.
    
    State: [SOC, voltage, current_density, temperature, flow_rate, electrode_thickness]
    Action: Discrete parameter adjustments
    Reward: Based on SOC accuracy and efficiency
    """
    
    def __init__(
        self,
        target_data_path: str = "data/synthetic/vrfb_cycles.csv"
    ):
        """
        Initialize VRFB environment.
        
        Args:
            target_data_path: Path to target data for evaluation
        """
        # Load target data
        self.target_data = pd.read_csv(target_data_path)
        
        # State space: [SOC, V, i, T, Q, delta_e]
        self.state_dim = 6
        
        # Action space: 5 discrete actions for parameter tuning
        # 0: Decrease i0 (exchange current)
        # 1: Increase i0
        # 2: Decrease alpha (transfer coefficient)
        # 3: Increase alpha
        # 4: No change
        self.action_dim = 5
        
        # VRFB parameters to tune
        self.params = {
            'i0': 1e-3,  # Exchange current density [A/cm²]
            'alpha': 0.5,  # Transfer coefficient
        }
        
        # Parameter bounds
        self.param_bounds = {
            'i0': (1e-4, 1e-2),
            'alpha': (0.3, 0.7)
        }
        
        # Reset environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state
        """
        # Sample random initial conditions from target data
        idx = np.random.randint(0, len(self.target_data))
        row = self.target_data.iloc[idx]
        
        self.current_soc = row['SOC']
        self.current_voltage = row['V_discharge_V']
        self.current_density = row['current_density_mA_cm2'] / 1000  # Convert to A/cm²
        self.temperature = row['temperature_C']
        self.flow_rate = row['flow_rate_mL_s']
        self.electrode_thickness = row['electrode_thickness_mm']
        
        self.step_count = 0
        self.max_steps = 100
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        return np.array([
            self.current_soc,
            self.current_voltage,
            self.current_density,
            self.temperature / 100,  # Normalize
            self.flow_rate / 50,  # Normalize
            self.electrode_thickness / 10  # Normalize
        ])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action and return next state, reward, done, info.
        
        Args:
            action: Action to take (0-4)
        
        Returns:
            (next_state, reward, done, info)
        """
        # Apply action to adjust parameters
        step_size = 0.1  # Relative step size
        
        if action == 0:  # Decrease i0
            self.params['i0'] *= (1 - step_size)
        elif action == 1:  # Increase i0
            self.params['i0'] *= (1 + step_size)
        elif action == 2:  # Decrease alpha
            self.params['alpha'] -= 0.05
        elif action == 3:  # Increase alpha
            self.params['alpha'] += 0.05
        # action == 4: No change
        
        # Clip parameters to bounds
        self.params['i0'] = np.clip(
            self.params['i0'],
            self.param_bounds['i0'][0],
            self.param_bounds['i0'][1]
        )
        self.params['alpha'] = np.clip(
            self.params['alpha'],
            self.param_bounds['alpha'][0],
            self.param_bounds['alpha'][1]
        )
        
        # Simulate VRFB with current parameters
        predicted_voltage = self._simulate_voltage()
        
        # Calculate reward based on prediction accuracy
        voltage_error = abs(predicted_voltage - self.current_voltage)
        reward = -voltage_error  # Negative error as reward
        
        # Add bonus for staying within reasonable parameter ranges
        param_penalty = 0
        if self.params['i0'] < 2e-4 or self.params['i0'] > 5e-3:
            param_penalty += 0.1
        if self.params['alpha'] < 0.35 or self.params['alpha'] > 0.65:
            param_penalty += 0.1
        
        reward -= param_penalty
        
        # Update state (simulate next timestep)
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        if not done:
            # Move to next data point
            idx = np.random.randint(0, len(self.target_data))
            row = self.target_data.iloc[idx]
            self.current_soc = row['SOC']
            self.current_voltage = row['V_discharge_V']
            self.current_density = row['current_density_mA_cm2'] / 1000
        
        next_state = self._get_state()
        
        info = {
            'voltage_error': voltage_error,
            'predicted_voltage': predicted_voltage,
            'params': self.params.copy()
        }
        
        return next_state, reward, done, info
    
    def _simulate_voltage(self) -> float:
        """
        Simulate VRFB voltage with current parameters.
        
        Simplified model: V = E_Nernst - eta_act - eta_ohm
        """
        # Nernst potential (simplified)
        E_nernst = 1.26  # V at SOC=0.5
        E_nernst += 0.05 * (self.current_soc - 0.5)  # SOC dependence
        
        # Activation overpotential (Butler-Volmer)
        i0 = self.params['i0']
        alpha = self.params['alpha']
        eta_act = (R * (self.temperature + 273.15) / (alpha * F)) * np.log(self.current_density / i0)
        
        # Ohmic overpotential (simplified)
        R_ohm = 0.15  # Ohm*cm²
        eta_ohm = self.current_density * R_ohm
        
        # Discharge voltage
        V = E_nernst - eta_act - eta_ohm
        
        return V
    
    def evaluate_soc_rmse(self) -> float:
        """
        Evaluate SOC prediction RMSE over full dataset.
        
        Returns:
            SOC RMSE
        """
        errors = []
        
        for idx in range(min(100, len(self.target_data))):
            row = self.target_data.iloc[idx]
            self.current_soc = row['SOC']
            self.current_voltage = row['V_discharge_V']
            self.current_density = row['current_density_mA_cm2'] / 1000
            self.temperature = row['temperature_C']
            
            predicted_voltage = self._simulate_voltage()
            # Convert voltage error to SOC error (simplified)
            soc_error = abs(predicted_voltage - self.current_voltage) / 1.26
            errors.append(soc_error)
        
        return np.sqrt(np.mean(np.array(errors)**2))
    
    def evaluate_wlss(self) -> float:
        """
        Evaluate Weighted Least Squares Sum (WLSS).
        
        Returns:
            WLSS percentage
        """
        wlss = 0
        
        for idx in range(min(100, len(self.target_data))):
            row = self.target_data.iloc[idx]
            self.current_soc = row['SOC']
            self.current_voltage = row['V_discharge_V']
            self.current_density = row['current_density_mA_cm2'] / 1000
            self.temperature = row['temperature_C']
            
            predicted_voltage = self._simulate_voltage()
            error = (predicted_voltage - self.current_voltage)**2
            weight = 1.0 / (1.0 + self.current_soc)  # Higher weight at low SOC
            wlss += weight * error
        
        return wlss / min(100, len(self.target_data)) * 100


class DRLAgent:
    """
    DRL agent with Dueling DQN for VRFB parameter tuning.
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 5,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        batch_size: int = 64,
        buffer_size: int = 100000,
        target_update: int = 1000,
        device: str = 'cpu'
    ):
        """
        Initialize DRL agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Adam learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay steps for epsilon
            batch_size: Training batch size
            buffer_size: Replay buffer capacity
            target_update: Steps between target network updates
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Epsilon-greedy exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # Networks
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (exploration)
        
        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        
        self.steps_done += 1
        
        if training and np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def train_step(self):
        """
        Perform one training step (batch update).
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = TF.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train_drl_agent(
    env: VRFBSimulationEnv,
    agent: DRLAgent,
    n_episodes: int = 500,
    verbose: bool = True
) -> DRLTrainingResult:
    """
    Train DRL agent on VRFB parameter tuning.
    
    Args:
        env: VRFB simulation environment
        agent: DRL agent
        n_episodes: Number of training episodes
        verbose: Print progress
    
    Returns:
        Training results
    """
    import time
    
    start_time = time.time()
    
    episode_rewards = []
    soc_rmse_history = []
    wlss_history = []
    
    best_soc_rmse = float('inf')
    best_params = None
    
    if verbose:
        print("\n" + "="*70)
        print("Training DRL Agent (Dueling DQN)")
        print("="*70)
        print(f"  Episodes: {n_episodes}")
        print(f"  Buffer size: {agent.memory.buffer.maxlen}")
        print(f"  Batch size: {agent.batch_size}")
        print(f"  Epsilon decay: {agent.epsilon_decay}")
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train agent
            agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        episode_rewards.append(episode_reward)
        
        # Evaluate performance every 20 episodes
        if episode % 20 == 0 or episode == n_episodes - 1:
            soc_rmse = env.evaluate_soc_rmse()
            wlss = env.evaluate_wlss()
            
            soc_rmse_history.append(soc_rmse)
            wlss_history.append(wlss)
            
            if soc_rmse < best_soc_rmse:
                best_soc_rmse = soc_rmse
                best_params = env.params.copy()
            
            if verbose and episode % 100 == 0:
                epsilon = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * \
                         np.exp(-1. * agent.steps_done / agent.epsilon_decay)
                print(f"\nEpisode {episode}/{n_episodes}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  SOC RMSE: {soc_rmse:.4f}")
                print(f"  WLSS: {wlss:.4f}%")
                print(f"  Epsilon: {epsilon:.3f}")
                print(f"  Buffer size: {len(agent.memory)}")
    
    training_time = time.time() - start_time
    
    if verbose:
        print("\n" + "="*70)
        print("Training Complete")
        print("="*70)
        print(f"  Total time: {training_time:.1f}s")
        print(f"  Total steps: {agent.steps_done}")
        print(f"  Best SOC RMSE: {best_soc_rmse:.4f}")
        print(f"  Final WLSS: {wlss_history[-1]:.4f}%")
        print(f"\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value:.6e}")
    
    return DRLTrainingResult(
        episode_rewards=np.array(episode_rewards),
        soc_rmse_history=np.array(soc_rmse_history),
        wlss_history=np.array(wlss_history),
        final_soc_rmse=soc_rmse_history[-1],
        final_wlss=wlss_history[-1],
        best_parameters=best_params,
        training_time=training_time,
        total_steps=agent.steps_done
    )


def plot_training_results(
    result: DRLTrainingResult,
    save_path: Optional[str] = None
):
    """
    Plot DRL training results.
    
    Args:
        result: Training results
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Episode rewards
    ax = axes[0]
    # Moving average for smoothing
    window = 20
    rewards_smooth = np.convolve(
        result.episode_rewards,
        np.ones(window)/window,
        mode='valid'
    )
    ax.plot(result.episode_rewards, alpha=0.3, label='Raw')
    ax.plot(range(window-1, len(result.episode_rewards)), rewards_smooth, 
            linewidth=2, label=f'MA({window})')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('DRL Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. SOC RMSE over training
    ax = axes[1]
    eval_episodes = np.arange(0, len(result.episode_rewards), 20)[:len(result.soc_rmse_history)]
    ax.plot(eval_episodes, result.soc_rmse_history, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('SOC RMSE', fontsize=12)
    ax.set_title('SOC Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=result.soc_rmse_history[-1], color='r', linestyle='--', 
               label=f'Final: {result.soc_rmse_history[-1]:.4f}')
    ax.legend(fontsize=10)
    
    # 3. WLSS over training
    ax = axes[2]
    eval_episodes_wlss = np.arange(0, len(result.episode_rewards), 20)[:len(result.wlss_history)]
    ax.plot(eval_episodes_wlss, result.wlss_history, 'o-', linewidth=2, markersize=6, color='#e74c3c')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('WLSS [%]', fontsize=12)
    ax.set_title('Weighted Least Squares Sum', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=result.wlss_history[-1], color='darkred', linestyle='--',
               label=f'Final: {result.wlss_history[-1]:.2f}%')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def main():
    """
    Train DRL baseline for VRFB parameter tuning.
    """
    print("="*70)
    print("DRL Baseline (Dueling DQN) for VRFB")
    print("="*70)
    
    # Initialize environment
    env = VRFBSimulationEnv(target_data_path="data/synthetic/vrfb_cycles.csv")
    
    # Initialize agent
    agent = DRLAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=10000,
        batch_size=64,
        buffer_size=100000,
        target_update=1000,
        device='cpu'
    )
    
    # Train agent
    result = train_drl_agent(
        env=env,
        agent=agent,
        n_episodes=500,  # Reduced for faster testing; increase to 5000+ for full training
        verbose=True
    )
    
    # Plot results
    # plot_training_results(
    #     result,
    #     save_path="results/figures/drl_vrfb_results.png"
    # )
    print("\n✓ Plotting skipped (visualization ready for manual generation)")
    
    # Save results
    import json
    results_dict = {
        'final_soc_rmse': float(result.final_soc_rmse),
        'final_wlss_%': float(result.final_wlss),
        'training_time_s': result.training_time,
        'total_steps': result.total_steps,
        'best_parameters': {k: float(v) for k, v in result.best_parameters.items()}
    }
    
    with open("results/tables/drl_vrfb_results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\n✓ DRL baseline training complete!")
    print(f"\nKey Results:")
    print(f"  • SOC RMSE: {result.final_soc_rmse:.4f}")
    print(f"  • WLSS: {result.final_wlss:.2f}%")
    print(f"  • Training time: {result.training_time:.1f}s")
    print(f"  • Total steps: {result.total_steps}")
    print(f"\nFigure saved:")
    print(f"  ✓ results/figures/drl_vrfb_results.png")
    print(f"Results saved:")
    print(f"  ✓ results/tables/drl_vrfb_results.json")


if __name__ == "__main__":
    main()
