import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import pickle
from feature_engineering import TransactionFeatures

class FraudDetectionNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout_rate=0.3):
        super(FraudDetectionNet, self).__init__()
        
        # Build neural network architecture
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class FraudDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize feature transformer
        self.feature_transformer = TransactionFeatures()
        
        # Model will be initialized after knowing input dimensions
        self.model = None
        self.criterion = nn.BCELoss()
        self.best_threshold = 0.5
        
        if model_path:
            self.load_model(model_path)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def prepare_data(self, data_file):
        """Load and prepare data for training"""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        # Load data
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        elif data_file.endswith('.pkl'):
            with open(data_file, 'rb') as f:
                transactions = pickle.load(f)
                df = pd.DataFrame(transactions)
        else:
            raise ValueError("Unsupported file format")
        
        print(f"✅ Loaded {len(df):,} transactions from {data_file}")
        print(f"   Fraud transactions: {df['isFraud'].sum():,}")
        print(f"   Normal transactions: {(~df['isFraud']).sum():,}")
        print(f"   Fraud rate: {df['isFraud'].mean():.2%}")
        print("=" * 60)
        
        # Fit feature transformer
        print("Fitting feature transformer...")
        self.feature_transformer.fit(df)
        
        # Transform all transactions - OPTIMIZED for large datasets
        print(f"Transforming all {len(df):,} transactions to features...")
        X = []
        y = []
        
        # Process in batches for progress reporting
        batch_size = 5000
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_df = df.iloc[i:batch_end]
            
            for _, row in batch_df.iterrows():
                features = self.feature_transformer.transform_transaction(row.to_dict())
                X.append(features)
                y.append(float(row['isFraud']))
            
            if (i + batch_size) % 10000 == 0 or batch_end == len(df):
                print(f"  Processed {batch_end:,}/{len(df):,} transactions ({100*batch_end/len(df):.1f}%)")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"✅ Feature transformation complete!")
        print(f"   Feature matrix shape: {X.shape}")
        print("=" * 60)
        
        # Initialize model with correct input dimensions
        if self.model is None:
            input_dim = X.shape[1]
            print(f"Initializing neural network with {input_dim} input features")
            self.model = FraudDetectionNet(
                input_dim=input_dim,
                hidden_dims=[128, 64, 32, 16],
                dropout_rate=0.3
            ).to(self.device)
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print("=" * 60)
        
        return X, y
    
    def create_weighted_sampler(self, y_train):
        """Create weighted sampler to handle class imbalance"""
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        weights = class_weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(
            weights=torch.FloatTensor(weights),
            num_samples=len(weights),
            replacement=True
        )
        return sampler
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, learning_rate=0.001):
        """Train the fraud detection model"""
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print("=" * 60)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loader with weighted sampling
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        sampler = self.create_weighted_sampler(y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        
        print(f"Number of batches per epoch: {len(train_loader)}")
        print("=" * 60)
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor).squeeze()
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                val_predicted = (val_outputs > 0.5).float()
                val_correct = (val_predicted == y_val_tensor).sum().item()
                val_accuracy = val_correct / len(y_val_tensor)
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Store history
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model('best_fraud_detector.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.load_model('best_fraud_detector.pth')
        
        # Find optimal threshold
        self.find_optimal_threshold(X_val, y_val)
        
        print("=" * 60)
        print("✅ Training complete!")
        print("=" * 60)
    
    def find_optimal_threshold(self, X_val, y_val):
        """Find optimal decision threshold using validation set"""
        print("\nFinding optimal decision threshold...")
        self.model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            val_probs = self.model(X_val_tensor).squeeze().cpu().numpy()
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.3, 0.8, 0.05):
            predictions = (val_probs > threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, predictions, average='binary', zero_division=0
            )
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.best_threshold = best_threshold
        print(f"✅ Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    def predict(self, transaction_features):
        """Predict fraud probability for a single transaction"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(transaction_features, dict):
                # Transform transaction to features
                features = self.feature_transformer.transform_transaction(transaction_features)
            else:
                features = transaction_features
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Get prediction
            prob = self.model(features_tensor).squeeze().cpu().item()
            
            return prob
    
    def predict_batch(self, X):
        """Predict fraud probabilities for a batch"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            probs = self.model(X_tensor).squeeze().cpu().numpy()
        return probs
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION ON TEST SET")
        print("=" * 60)
        print(f"Test samples: {len(X_test):,}")
        
        # Get predictions
        probs = self.predict_batch(X_test)
        predictions = (probs > self.best_threshold).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='binary', zero_division=0
        )
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test, probs)
        except:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Calculate accuracy
        accuracy = np.sum(predictions == y_test) / len(y_test)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        print(f"\n📊 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"               Normal  Fraud")
        print(f"   Actual Normal  {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"   Actual Fraud   {cm[1,0]:5d}  {cm[1,1]:5d}")
        print("=" * 60)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Acc')
        ax2.plot(self.val_accuracies, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("📊 Training history plot saved to 'training_history.png'")
        plt.show()
    
    def save_model(self, path):
        """Save model and feature transformer"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_transformer': self.feature_transformer,
            'best_threshold': self.best_threshold,
            'feature_names': self.feature_transformer.feature_names
        }, path)
    
    def load_model(self, path):
        """Load model and feature transformer"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load feature transformer
        self.feature_transformer = checkpoint['feature_transformer']
        
        # Initialize model if not already done
        if self.model is None:
            input_dim = len(self.feature_transformer.feature_names)
            self.model = FraudDetectionNet(
                input_dim=input_dim,
                hidden_dims=[128, 64, 32, 16]
            ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_threshold = checkpoint.get('best_threshold', 0.5)

# Main training script
if __name__ == "__main__":
    import glob
    import os
    
    print("=" * 60)
    print("FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Find the most recent data file
    csv_files = glob.glob("fraud_transactions_*.csv")
    pkl_files = glob.glob("fraud_transactions_*.pkl")
    
    if pkl_files:
        data_file = max(pkl_files, key=os.path.getctime)
        print("Found pickle files (faster loading)")
    elif csv_files:
        data_file = max(csv_files, key=os.path.getctime)
        print("Found CSV files")
    else:
        print("❌ No data files found! Run fraud_collector.py first.")
        exit(1)
    
    print(f"Using most recent file: {data_file}")
    print("=" * 60)
    
    # Create detector
    detector = FraudDetector()
    
    # Prepare data - THIS WILL USE ALL 50,000 TRANSACTIONS
    X, y = detector.prepare_data(data_file)
    
    print("\nSplitting dataset...")
    # Split data: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )  # 0.176 * 0.85 ≈ 0.15 for validation
    
    print(f"\n📊 Dataset splits:")
    print(f"   Train: {len(X_train):,} samples ({100*len(X_train)/len(X):.1f}%) - Fraud rate: {y_train.mean():.2%}")
    print(f"   Val:   {len(X_val):,} samples ({100*len(X_val)/len(X):.1f}%) - Fraud rate: {y_val.mean():.2%}")
    print(f"   Test:  {len(X_test):,} samples ({100*len(X_test)/len(X):.1f}%) - Fraud rate: {y_test.mean():.2%}")
    print(f"   TOTAL: {len(X):,} transactions being used for training!")
    
    # Train model with larger batch size for 50K dataset
    detector.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=128)
    
    # Evaluate on test set
    metrics = detector.evaluate(X_test, y_test)
    
    # Plot training history
    detector.plot_training_history()
    
    # Save final model
    detector.save_model('fraud_detector.pth')
    print(f"\n💾 Final model saved to 'fraud_detector.pth'")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total transactions used: {len(X):,}")
    print(f"Final F1-Score: {metrics['f1']:.4f}")
    print(f"Final ROC-AUC: {metrics['roc_auc']:.4f}")
    print("=" * 60)
