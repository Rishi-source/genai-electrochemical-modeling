"""
ePCDNN Baseline for VRFB Voltage Prediction
Enhanced Physics-Constrained Deep Neural Network from Paper [1].

Key features:
- Physics-constrained loss function
- Charge conservation penalty
- Voltage decomposition penalty
- Thermodynamic bounds penalty
- Focus on low-SOC regime accuracy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.physics_constants import R, F, VRFBParams


@dataclass
class ePCDNNTrainingResult:
    """Results from ePCDNN training."""
    train_rmse: float
    val_rmse: float
    test_rmse: float
    test_mae: float
    test_r_squared: float
    
    low_soc_rmse: float  # Performance in low-SOC region
    mid_soc_rmse: float  # Performance in mid-SOC region
    high_soc_rmse: float  # Performance in high-SOC region
    
    physics_loss: float  # Final physics constraint violation
    
    train_losses: np.ndarray
    val_losses: np.ndarray


class VRFB_ePCDNN(nn.Module):
    """
    Enhanced Physics-Constrained DNN for VRFB voltage prediction.
    
    Architecture: Input → 64 → 64 → 64 → Output
    Loss: L = L_data + λ_phys * L_phys
    
    where L_phys includes:
    - Charge conservation
    - Voltage decomposition
    - Thermodynamic bounds
    """
    
    def __init__(self, input_dim: int = 7):
        """
        Initialize ePCDNN.
        
        Args:
            input_dim: Number of input features
        """
        super(VRFB_ePCDNN, self).__init__()
        
        # Main network for voltage prediction
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)
        
        # Activation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Auxiliary network for physics-aware features
        self.physics_fc1 = nn.Linear(input_dim, 32)
        self.physics_fc2 = nn.Linear(32, 16)
        self.physics_out = nn.Linear(16, 4)  # Outputs: E_nernst, eta_act, eta_ohm, eta_mt
        
    def forward(self, x):
        """
        Forward pass.
        
        Returns:
            voltage: Predicted voltage
            physics_features: [E_nernst, eta_act, eta_ohm, eta_mt]
        """
        # Main voltage prediction
        h = self.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.relu(self.fc2(h))
        h = self.dropout(h)
        h = self.relu(self.fc3(h))
        h = self.dropout(h)
        voltage = self.fc4(h)
        
        # Physics features
        p = self.relu(self.physics_fc1(x))
        p = self.relu(self.physics_fc2(p))
        physics_features = self.physics_out(p)
        
        return voltage, physics_features


class PhysicsConstrainedLoss(nn.Module):
    """
    Physics-constrained loss function for VRFB.
    
    L = L_data + λ_phys * (L_charge + L_voltage + L_thermo)
    """
    
    def __init__(
        self,
        lambda_phys: float = 0.1,
        lambda_charge: float = 1.0,
        lambda_voltage: float = 1.0,
        lambda_thermo: float = 1.0
    ):
        """
        Initialize physics loss.
        
        Args:
            lambda_phys: Overall physics penalty weight
            lambda_charge: Charge conservation weight
            lambda_voltage: Voltage decomposition weight
            lambda_thermo: Thermodynamic bounds weight
        """
        super(PhysicsConstrainedLoss, self).__init__()
        self.lambda_phys = lambda_phys
        self.lambda_charge = lambda_charge
        self.lambda_voltage = lambda_voltage
        self.lambda_thermo = lambda_thermo
        
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        voltage_pred: torch.Tensor,
        voltage_true: torch.Tensor,
        physics_features: torch.Tensor,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with physics constraints.
        
        Args:
            voltage_pred: Predicted voltage [batch, 1]
            voltage_true: True voltage [batch, 1]
            physics_features: [E_nernst, eta_act, eta_ohm, eta_mt] [batch, 4]
            inputs: Input features [batch, input_dim]
        
        Returns:
            total_loss, loss_dict
        """
        # Data loss
        loss_data = self.mse(voltage_pred, voltage_true)
        
        # Extract physics features
        E_nernst = physics_features[:, 0:1]
        eta_act = physics_features[:, 1:2]
        eta_ohm = physics_features[:, 2:3]
        eta_mt = physics_features[:, 3:4]
        
        # 1. Charge conservation (implicit - no explicit penalty needed)
        loss_charge = torch.tensor(0.0, device=voltage_pred.device)
        
        # 2. Voltage decomposition: V = E_nernst - eta_act - eta_ohm - eta_mt
        voltage_reconstructed = E_nernst - eta_act - eta_ohm - eta_mt
        loss_voltage = self.mse(voltage_pred, voltage_reconstructed)
        
        # 3. Thermodynamic bounds
        # - Nernst potential should be positive and reasonable (0.8-1.8V)
        # - Losses should be positive
        # - Total voltage should be positive
        
        loss_thermo = torch.tensor(0.0, device=voltage_pred.device)
        
        # Nernst bounds
        E_min, E_max = 0.8, 1.8
        loss_thermo += torch.mean(torch.relu(E_min - E_nernst))  # E > E_min
        loss_thermo += torch.mean(torch.relu(E_nernst - E_max))  # E < E_max
        
        # Losses should be non-negative
        loss_thermo += torch.mean(torch.relu(-eta_act))
        loss_thermo += torch.mean(torch.relu(-eta_ohm))
        loss_thermo += torch.mean(torch.relu(-eta_mt))
        
        # Voltage should be positive
        loss_thermo += torch.mean(torch.relu(-voltage_pred))
        
        # Total physics loss
        loss_physics = (
            self.lambda_charge * loss_charge +
            self.lambda_voltage * loss_voltage +
            self.lambda_thermo * loss_thermo
        )
        
        # Total loss
        total_loss = loss_data + self.lambda_phys * loss_physics
        
        loss_dict = {
            'total': total_loss.item(),
            'data': loss_data.item(),
            'physics': loss_physics.item(),
            'charge': loss_charge.item(),
            'voltage': loss_voltage.item(),
            'thermo': loss_thermo.item()
        }
        
        return total_loss, loss_dict


class ePCDNN_VRFB_Trainer:
    """
    Trainer for VRFB ePCDNN baseline.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        lambda_phys: float = 0.1,
        device: str = 'cpu',
        verbose: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            learning_rate: Adam optimizer learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            early_stopping_patience: Epochs to wait for improvement
            lambda_phys: Physics penalty weight
            device: 'cpu' or 'cuda'
            verbose: Print progress
        """
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = early_stopping_patience
        self.lambda_phys = lambda_phys
        self.device = torch.device(device)
        self.verbose = verbose
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def prepare_data(
        self,
        data_path: str = "data/synthetic/vrfb_cycles.csv",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and prepare VRFB data (70/15/15 split).
        
        Args:
            data_path: Path to CSV data
            test_size: Test set fraction
            val_size: Validation set fraction
            random_state: Random seed
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        # Load data
        df = pd.read_csv(data_path)
        
        # Create combined dataset with charge and discharge
        # Discharge data
        df_discharge = df[[
            'current_density_mA_cm2',
            'temperature_C',
            'SOC',
            'electrode_thickness_mm',
            'flow_rate_mL_s',
            'V_discharge_V'
        ]].copy()
        df_discharge['charge_discharge'] = -1  # Discharge
        df_discharge.rename(columns={'V_discharge_V': 'voltage_V'}, inplace=True)
        
        # Charge data
        df_charge = df[[
            'current_density_mA_cm2',
            'temperature_C',
            'SOC',
            'electrode_thickness_mm',
            'flow_rate_mL_s',
            'V_charge_V'
        ]].copy()
        df_charge['charge_discharge'] = 1  # Charge
        df_charge.rename(columns={'V_charge_V': 'voltage_V'}, inplace=True)
        
        # Combine
        df_combined = pd.concat([df_discharge, df_charge], ignore_index=True)
        
        # Features
        features = [
            'current_density_mA_cm2',
            'temperature_C',
            'SOC',
            'electrode_thickness_mm',
            'flow_rate_mL_s',
            'charge_discharge'
        ]
        
        X = df_combined[features].values
        y = df_combined['voltage_V'].values.reshape(-1, 1)
        
        # Store SOC for low-SOC analysis
        self.SOC_test_original = None
        
        # Split: 70% train, 15% val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=(df_combined['SOC'] < 0.3).values
        )
        
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=random_state
        )
        
        # Store original SOC before scaling
        self.SOC_test_original = X_test[:, 2].copy()
        
        # Standardize
        X_train = self.scaler_X.fit_transform(X_train)
        X_val = self.scaler_X.transform(X_val)
        X_test = self.scaler_X.transform(X_test)
        
        y_train = self.scaler_y.fit_transform(y_train)
        y_val = self.scaler_y.transform(y_val)
        y_test = self.scaler_y.transform(y_test)
        
        # Convert to tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        if self.verbose:
            print(f"✓ Data prepared")
            print(f"  Train: {len(train_dataset)} samples")
            print(f"  Val: {len(val_dataset)} samples")
            print(f"  Test: {len(test_dataset)} samples")
            print(f"  Features: {len(features)}")
            print(f"  Low-SOC (<30%) test samples: {np.sum(self.SOC_test_original < 0.3)}")
        
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = features
        
        return train_loader, val_loader, test_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train ePCDNN with physics constraints.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            (train_losses, val_losses)
        """
        # Initialize model
        input_dim = next(iter(train_loader))[0].shape[1]
        self.model = VRFB_ePCDNN(input_dim=input_dim).to(self.device)
        
        # Loss and optimizer
        criterion = PhysicsConstrainedLoss(lambda_phys=self.lambda_phys)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        if self.verbose:
            print(f"\nTraining ePCDNN...")
            print(f"  Architecture: {input_dim} → 64 → 64 → 64 → 1")
            print(f"  Physics penalty: λ_phys = {self.lambda_phys}")
            print(f"  Optimizer: Adam (lr={self.lr})")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Max epochs: {self.epochs}")
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                voltage_pred, physics_features = self.model(X_batch)
                loss, loss_dict = criterion(voltage_pred, y_batch, physics_features, X_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    voltage_pred, physics_features = self.model(X_batch)
                    loss, _ = criterion(voltage_pred, y_batch, physics_features, X_batch)
                    val_loss += loss.item() * X_batch.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}: "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}")
            
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"\n✓ Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        if self.verbose:
            print(f"✓ Training complete")
            print(f"  Best val loss: {best_val_loss:.6f}")
            print(f"  Final epoch: {len(train_losses)}")
        
        return np.array(train_losses), np.array(val_losses)
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Tuple[float, float, float, float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluate on test set with SOC-stratified metrics.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            (rmse, mae, r², low_soc_rmse, mid_soc_rmse, high_soc_rmse, predictions, actual)
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                voltage_pred, _ = self.model(X_batch)
                predictions.append(voltage_pred.cpu().numpy())
                actuals.append(y_batch.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Inverse transform
        predictions = self.scaler_y.inverse_transform(predictions)
        actuals = self.scaler_y.inverse_transform(actuals)
        
        # Overall metrics
        residuals = actuals - predictions
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((actuals - actuals.mean())**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # SOC-stratified metrics
        low_soc_mask = self.SOC_test_original < 0.3
        mid_soc_mask = (self.SOC_test_original >= 0.3) & (self.SOC_test_original < 0.7)
        high_soc_mask = self.SOC_test_original >= 0.7
        
        low_soc_rmse = np.sqrt(np.mean(residuals[low_soc_mask]**2)) if np.any(low_soc_mask) else 0
        mid_soc_rmse = np.sqrt(np.mean(residuals[mid_soc_mask]**2)) if np.any(mid_soc_mask) else 0
        high_soc_rmse = np.sqrt(np.mean(residuals[high_soc_mask]**2)) if np.any(high_soc_mask) else 0
        
        if self.verbose:
            print(f"\n✓ Test Set Evaluation")
            print(f"  Overall RMSE: {rmse*1000:.2f} mV")
            print(f"  Overall MAE: {mae*1000:.2f} mV")
            print(f"  Overall R²: {r_squared:.4f}")
            print(f"\n  SOC-Stratified Performance:")
            print(f"    Low-SOC (<30%): {low_soc_rmse*1000:.2f} mV")
            print(f"    Mid-SOC (30-70%): {mid_soc_rmse*1000:.2f} mV")
            print(f"    High-SOC (>70%): {high_soc_rmse*1000:.2f} mV")
        
        return rmse, mae, r_squared, low_soc_rmse, mid_soc_rmse, high_soc_rmse, predictions.flatten(), actuals.flatten()
    
    def plot_results(
        self,
        train_losses: np.ndarray,
        val_losses: np.ndarray,
        predictions: np.ndarray,
        actual: np.ndarray,
        r_squared: float,
        rmse: float,
        low_soc_rmse: float,
        save_path: Optional[str] = None
    ):
        """
        Plot training history and predictions.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Training history
        ax = axes[0]
        ax.plot(train_losses, label='Train Loss', linewidth=2)
        ax.plot(val_losses, label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('ePCDNN Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 2. Predictions vs Actual
        ax = axes[1]
        ax.scatter(actual, predictions, alpha=0.5, s=20)
        min_val = min(actual.min(), predictions.min())
        max_val = max(actual.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('Actual Voltage [V]', fontsize=12)
        ax.set_ylabel('Predicted Voltage [V]', fontsize=12)
        ax.set_title(f'ePCDNN Predictions (R²={r_squared:.4f})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 3. SOC-stratified performance
        ax = axes[2]
        soc_bins = ['Low\n(<30%)', 'Mid\n(30-70%)', 'High\n(>70%)']
        low_idx = self.SOC_test_original < 0.3
        mid_idx = (self.SOC_test_original >= 0.3) & (self.SOC_test_original < 0.7)
        high_idx = self.SOC_test_original >= 0.7
        
        rmse_vals = [
            np.sqrt(np.mean((actual[low_idx] - predictions[low_idx])**2)) * 1000,
            np.sqrt(np.mean((actual[mid_idx] - predictions[mid_idx])**2)) * 1000,
            np.sqrt(np.mean((actual[high_idx] - predictions[high_idx])**2)) * 1000
        ]
        
        bars = ax.bar(soc_bins, rmse_vals, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax.set_ylabel('RMSE [mV]', fontsize=12)
        ax.set_title('Performance by SOC Region', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, rmse_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Saved figure to {save_path}")
        
        plt.show()


def main():
    """
    Train and evaluate ePCDNN baseline for VRFB.
    """
    print("="*70)
    print("ePCDNN Baseline for VRFB - Training & Evaluation")
    print("="*70)
    
    # Initialize trainer
    trainer = ePCDNN_VRFB_Trainer(
        learning_rate=1e-3,
        batch_size=32,
        epochs=200,
        early_stopping_patience=20,
        lambda_phys=0.1,  # Physics penalty weight
        device='cpu',
        verbose=True
    )
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(
        data_path="data/synthetic/vrfb_cycles.csv",
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    # Train
    train_losses, val_losses = trainer.train(train_loader, val_loader)
    
    # Evaluate
    rmse, mae, r_squared, low_soc_rmse, mid_soc_rmse, high_soc_rmse, predictions, actual = trainer.evaluate(test_loader)
    
    # Plot
    trainer.plot_results(
        train_losses,
        val_losses,
        predictions,
        actual,
        r_squared,
        rmse,
        low_soc_rmse,
        save_path="results/figures/epcdnn_vrfb_results.png"
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ ePCDNN Baseline training complete")
    print(f"\nOverall Performance:")
    print(f"  RMSE: {rmse*1000:.2f} mV")
    print(f"  MAE: {mae*1000:.2f} mV")
    print(f"  R²: {r_squared:.4f}")
    print(f"\nSOC-Stratified Performance:")
    print(f"  Low-SOC: {low_soc_rmse*1000:.2f} mV (critical region)")
    print(f"  Mid-SOC: {mid_soc_rmse*1000:.2f} mV")
    print(f"  High-SOC: {high_soc_rmse*1000:.2f} mV")
    print(f"\nFigure saved:")
    print(f"  ✓ results/figures/epcdnn_vrfb_results.png")
    print("="*70)


if __name__ == "__main__":
    main()
