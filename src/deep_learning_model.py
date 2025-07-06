import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

from evaluation import evaluate_classification_model
from wandb_utils import initialize_wandb_run, log_wandb_artifact, log_wandb_metrics, finish_wandb_run



## --- 1. Define the Neural Network Architecture
class BankMarketingNN(nn.Module):
    def __init__(self, input_dim:int, hidden_layers: list, dropout_rate: float= 0.3):
        """
        Initialize a simple Feed-Forward NN for Binary classification.
        
        Args:
            input_dim (int): Number of input features
            hidden_layer (list): a list of integers, where each integer represents the number of neurons in a hidden layer
            dropout_rate (float):  Dropout probal for regularization
        """
        
        super(BankMarketingNN, self).__init__()
        
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))  # Fully connected layer
            layers.append(nn.ReLU())                      # ReLU activation
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        
        # Output layer: 1 neuron for Binary classification (logits)
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        
    def forward(self, x):
        """
        Forward pass through the netward
        Args:
            X (torch.Tensor): Input tensor
        Returns
            torch.Tensor : output logits
        
        """
        return self.network(x)
    
    
# --- 2. Training and Eval function
def train_deeL_model(
    X_Train: np.ndarray,
    y_train: np.ndarray,
    X_Test: np.ndarray,
    y_test:np.ndarray,
    dl_config: dict,
    run_name_suffix: str= "",
    wandb_project_name: str = "Bank-Marketing-ML-Comparison",
    output_dir: str = "models/deep_learningModel",
    device: str = "cuda"  if torch.cuda.is_available() else "cpu"
):   
    
    """
    Trains and evaluate the DL model, logging results into Weight & Biases
    
    Args:
        X_train (np.ndarray): Preprocessed training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarrray): Preprocessed test features
        y_test (np.ndarray) : Test labels
        dL_config (dict) : Dict of DL model hyperparameters
        run_name_suffix (str): Suffic to add to the W&B run name for distinction
        wandb_project_name (str): Name of the W&B project,
        output_dir (str): Dict to save the trained model 
        device (str): Device to run the training on ('cuda' or 'cpu')        
    Returns:
        BandMarketingNN: The trained Pytorch model
    """    
    
    print(f"\n-- Training Deep Learning Model on {device} --")
    
    # Initialize W&B run
    run_name = f"DeepLearningNN-{run_name_suffix}" if run_name_suffix else "DeepLearningNN"
    wandb_run = initialize_wandb_run(
        project_name=wandb_project_name,
        run_name=run_name,
        config={"model_type":"DeepLearningNN", **dl_config}
    )
    
    # Convert numpy arrays to Torch tensor
    X_train_tensor = torch.tensor(X_Train, dtype=torch.float32).to(device)    
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_Test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Create TensorDataset and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size = dl_config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = dl_config['batch_size'], shuffle=False)
    
    # Initialize model
    input_dim = X_Train.shape[1]
    model = BankMarketingNN(input_dim= input_dim,
                            hidden_layers=dl_config['hidden_layers'],
                            dropout_rate=dl_config['dropout_rate']).to(device)
    
    # Define the loss fct and optimzer
    # BCEWithLogitsLoww combines sigmoid and BCE
    # pos_weight helps handle class imbalance by weighting the positive class loss
    # Calculate pos_weight: count_negative/count_positive
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32).to(device)
    print(f"Calculated positive class weight for BCEWithLogistsLoss: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = None
    if dl_config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=dl_config['learning_rate'])
    elif dl_config['optimizer'] ==  'SGD':
        optimizer = optim.SGD(model.parameters(), lr=dl_config['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {dl_config['optimizer']}")

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = 0
    
    
    # Training loop
    print(f"🏋️-- Starting Training for {dl_config['num_epochs']} epochs ---")
    for epoch in range(dl_config['num_epochs']):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probas = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch +1} Validation ✓"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)  
                
                
                # convert logits to proba
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probas.extend(probabilities.cpu().numpy())
                
        val_loss /= len(test_loader.dataset)
          
        # Flatten list for eval metric
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        all_probas = np.array(all_probas).flatten()
        
        # Evaluate using evalu module
        val_metrics = evaluate_classification_model(
            y_true = all_labels,
            y_pred =  all_preds,
            y_proba = all_probas,
            model_name='DL_Validation',
            plot_result=False,
            wandb_log=False            
        )
        
        
        print(f"Epoch {epoch+1}/{dl_config['num_epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy {val_metrics['accuracy']:.4f} ,  Val F1-Score: {val_metrics['f1_score']:.4f}")
      
        log_wandb_metrics({
         "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics['accuracy'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_f1_score": val_metrics['f1_score'],
            "val_roc_auc": val_metrics['roc_auc']
        }, step=epoch)
    
        # Eearly stopping check
        print(type(val_loss),type(best_model_state), "--------------")
        if (val_loss) < (best_val_loss):
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict() # Save best model state
            print("Validation loss improved. Saving best model state")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{dl_config['early_stopping_patience']}")
            if patience_counter >= dl_config['early_stopping_patience']:
                print(f"Early stopping triggered at each epoch {epoch+1}")
                break
            
        
    # Load the best model state if early stopping occured
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model state for final evaluation")
    else:
        print("No improvement observed, using model from last epoch")
    
    
    # -- Final Evaluation of tes set
    print("\n 🤔-- Final Evaluation of DL test set --")
    model.eval()
    final_preds = []
    final_labels = []
    final_probas = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Final test evaluation"):
            outputs =  model(inputs)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            
            final_preds.extend(predictions.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
            final_probas.extend(probabilities.cpu().numpy())
    
    final_preds = np.array(final_preds).flatten()
    final_labels = np.array(final_labels).flatten()
    final_probas = np.array(final_probas).flatten()
    
    final_test_metrics = evaluate_classification_model(
         y_true=final_labels,
        y_pred=final_preds,
        y_proba=final_probas,
        model_name="DeepLearningNN_Test",
        plot_results=True, # Show plots for final evaluation
        wandb_log=True     # Log plots to W&B
    )
    
        # Log final test metrics to W&B
    log_wandb_metrics({f"test_{k}": v for k, v in final_test_metrics.items()})

    # Save the trained PyTorch model locally
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"deep_learning_nn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Deep Learning model saved locally to: {model_path}")

    # Log the saved model as a W&B artifact
    log_wandb_artifact(
        artifact_name="DeepLearningNN_Model",
        artifact_type="model",
        file_path=model_path,
        description="Trained PyTorch Deep Learning Model"
    )

    # Finish the W&B run
    finish_wandb_run()

    return model


if __name__ == "__main__":

    # Example of Usage
    print("Running standalone test of DLmodel using dummy data")
    
    # Dummy data to tes model is functioning
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Crete a dummy Dataset with some categorical features
    X_dummy, y_dummy = make_classification(
        n_samples=500, n_features=20, n_informative=10, n_redundant=0,
        n_classes=2, n_clusters_per_class=1, weights=[0.9, 0.1], random_state=42
    )
    
    
    # Simulate preprocessing
    scaler = StandardScaler()
    X_dummy_scaled = scaler.fit_transform(X_dummy)
    
    X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(
        X_dummy_scaled, y_dummy, test_size=0.3, random_state=42, stratify=y_dummy
    )
    
    # Dummy config
    dl_config_dummy = {
        'input_dim': X_train_dummy.shape[1],
        'hidden_layers':[64, 32],
        'dropout_rate': 0.2,
        'num_epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'loss_function': 'BCEWithLogitsLoss',
        'early_stopping_patience': 5,
        'class_weight_dl': [1.0, 9.0]
    }
    
    try:
        trained_dl_model = train_deeL_model(
            X_Train=X_train_dummy,
            y_train=y_train_dummy,
            X_Test=X_test_dummy,
            y_test=y_test_dummy, 
            dl_config=dl_config_dummy,
            run_name_suffix="dummy-tes-run",
            wandb_project_name="deep-learning-models-test"
        )
        print("Deep learning model trained: {trained_dl_model}")
    except Exception as e:
        print(f"Error during DeeL model test: {e}")
    finally:
        pass
    