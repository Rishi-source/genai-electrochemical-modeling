import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@dataclass
class ANNTrainingResult:
    train_rmse: float
    val_rmse: float
    test_rmse: float
    test_mae: float
    test_r_squared: float
    
    train_losses: np.ndarray
    val_losses: np.ndarray
    
    predictions: np.ndarray
    actual: np.ndarray
    
    feature_importance: Optional[np.ndarray] = None


class PEMFC_ANN(nn.Module):
    
    def __init__(self, input_dim: int = 5):
        super(PEMFC_ANN, self).__init__()
        
        
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class PEMFC_ANN_Trainer:
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        device: str = 'cpu',
        verbose: bool = True
    ):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = early_stopping_patience
        self.device = torch.device(device)
        self.verbose = verbose
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def prepare_data(
        self,
        data_path: str = "data/synthetic/pemfc_polarization.csv",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        df = pd.read_csv(data_path)
        
        
        features = [
            'current_density_A_cm2',
            'temperature_C',
            'stoich_anode',
            'stoich_cathode',
            'p_H2_atm'  
        ]
        
        X = df[features].values
        y = df['voltage_V'].values.reshape(-1, 1)
        
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=random_state
        )
        
        
        X_train = self.scaler_X.fit_transform(X_train)
        X_val = self.scaler_X.transform(X_val)
        X_test = self.scaler_X.transform(X_test)
        
        y_train = self.scaler_y.fit_transform(y_train)
        y_val = self.scaler_y.transform(y_val)
        y_test = self.scaler_y.transform(y_test)
        
        
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
        
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        if self.verbose:
            print(f"✓ Data prepared")
            print(f"  Train: {len(train_dataset)} samples")
            print(f"  Val: {len(val_dataset)} samples")
            print(f"  Test: {len(test_dataset)} samples")
            print(f"  Features: {len(features)}")
        
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = features
        
        return train_loader, val_loader, test_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> ANNTrainingResult:
        
        input_dim = next(iter(train_loader))[0].shape[1]
        self.model = PEMFC_ANN(input_dim=input_dim).to(self.device)
        
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        if self.verbose:
            print(f"\n Training ANN...")
            print(f"  Architecture: {input_dim} → 50 → 50 → 50 → 1")
            print(f"  Optimizer: Adam (lr={self.lr})")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Max epochs: {self.epochs}")
            print(f"  Early stopping patience: {self.patience}")
        
        for epoch in range(self.epochs):
            
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            
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
        
        
        self.model.load_state_dict(best_model_state)
        
        if self.verbose:
            print(f"✓ Training complete")
            print(f"  Best val loss: {best_val_loss:.6f}")
            print(f"  Final epoch: {len(train_losses)}")
        
        return train_losses, val_losses
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.append(outputs.cpu().numpy())
                actuals.append(y_batch.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        
        predictions = self.scaler_y.inverse_transform(predictions)
        actuals = self.scaler_y.inverse_transform(actuals)
        
        
        residuals = actuals - predictions
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((actuals - actuals.mean())**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        if self.verbose:
            print(f"\n✓ Test Set Evaluation")
            print(f"  RMSE: {rmse*1000:.2f} mV")
            print(f"  MAE: {mae*1000:.2f} mV")
            print(f"  R²: {r_squared:.4f}")
        
        return rmse, mae, r_squared, predictions.flatten(), actuals.flatten()
    
    def compute_shap_importance(
        self,
        n_samples: int = 100
    ) -> np.ndarray:
        if self.verbose:
            print(f"\nComputing SHAP feature importance...")
        
        
        indices = np.random.choice(len(self.X_test), min(n_samples, len(self.X_test)), replace=False)
        X_sample = self.X_test[indices]
        
        
        def model_predict(x):
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(self.device)
                return self.model(x_tensor).cpu().numpy()
        
        
        explainer = shap.KernelExplainer(model_predict, X_sample)
        shap_values = explainer.shap_values(X_sample)
        
        
        importance = np.abs(shap_values).mean(axis=0)
        
        if self.verbose:
            print(f"✓ SHAP importance computed")
            for i, (name, imp) in enumerate(zip(self.feature_names, importance)):
                print(f"  {name}: {imp:.4f}")
        
        return importance
    
    def plot_results(
        self,
        train_losses: np.ndarray,
        val_losses: np.ndarray,
        predictions: np.ndarray,
        actual: np.ndarray,
        r_squared: float,
        rmse: float,
        save_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        
        ax = axes[0]
        ax.plot(train_losses, label='Train Loss', linewidth=2)
        ax.plot(val_losses, label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        
        ax = axes[1]
        ax.scatter(actual, predictions, alpha=0.5, s=20)
        
        
        min_val = min(actual.min(), predictions.min())
        max_val = max(actual.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel('Actual Voltage [V]', fontsize=12)
        ax.set_ylabel('Predicted Voltage [V]', fontsize=12)
        ax.set_title(f'ANN Predictions (R²={r_squared:.4f}, RMSE={rmse*1000:.2f}mV)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Saved figure to {save_path}")
        
        plt.show()


def main():
    print("="*70)
    print("ANN Baseline for PEMFC - Training & Evaluation")
    print("="*70)
    
    
    trainer = PEMFC_ANN_Trainer(
        learning_rate=1e-3,
        batch_size=32,
        epochs=200,
        early_stopping_patience=20,
        device='cpu',
        verbose=True
    )
    
    
    train_loader, val_loader, test_loader = trainer.prepare_data(
        data_path="data/synthetic/pemfc_polarization.csv",
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    
    train_losses, val_losses = trainer.train(train_loader, val_loader)
    
    
    rmse, mae, r_squared, predictions, actual = trainer.evaluate(test_loader)
    
    
    try:
        importance = trainer.compute_shap_importance(n_samples=100)
    except Exception as e:
        print(f"⚠ SHAP computation failed: {e}")
        importance = None
    
    
    trainer.plot_results(
        np.array(train_losses),
        np.array(val_losses),
        predictions,
        actual,
        r_squared,
        rmse,
        save_path="results/figures/ann_pemfc_results.png"
    )
    
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ ANN Baseline training complete")
    print(f"\nTest Performance:")
    print(f"  RMSE: {rmse*1000:.2f} mV")
    print(f"  MAE: {mae*1000:.2f} mV")
    print(f"  R²: {r_squared:.4f}")
    print(f"\nFigure saved:")
    print(f"  ✓ results/figures/ann_pemfc_results.png")
    print("="*70)


if __name__ == "__main__":
    main()
