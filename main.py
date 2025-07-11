import os
import yaml
import pandas as pd
import numpy as np
import torch

from src.data_pipeline import get_processed_data
from src.classical_models import train_classical_model
from src.deep_learning_model import train_deeL_model
from src.wandb_utils import finish_wandb_run

from src.wandb_utils import finish_wandb_run


def main():
    """
        Main function to orchestrate the entire ML pipeline
        data leading, cleaning, preprocessing, classical model training & evaluation
        and deep learning model training& evaluation
    """
    print("--- Starting Bank Marketing ML Comparison Project ---")
    
    
    # --- 1. Load Configuration
    config_path = 'configs/model_configs.yaml'
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Configuration loaded from {config_path}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return 
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return
    
    project_name = config['project_name']
    random_state =  config['random_state']
    test_size = config['test_size']
    
    numerical_scaler_type = config['numerical_scaler_type']
    apply_smote_classical = config['apply_smote']
    
    # Device for PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch will run on device: {device}") 
    
    
    # --- 2. Data Pipelinee
    print("Preparing data for Classical Models (with SMOTE if enabled) ...")
    X_train_classical, X_test_classical, y_train_classical, y_test_classical, full_pipeline_classical, feature_name_classical = get_processed_data(
        file_path='data/bank-additional-full.csv',
        test_size=test_size,
        random_state=random_state,
        apply_smote=apply_smote_classical,
        smote_sampling_strategy=True,
        numerical_scaler_type=numerical_scaler_type
    )
    
    
    # Extract the ColumnTransformer part of the pipeline for classical models
    preprocessor_for_classical = full_pipeline_classical.named_steps['preprocessor']
    
    
    ## Data for Deep Learning Model (no SMOTE, rely on class weights in loss function)
    # We will use the processed numpy array for PyTorch
    print("\n Preparing data for Deep learning Model (without SMOTE, using class weights) ")
    X_train_dl, X_test_dl, y_train_dl, y_test_dl, full_pipeline_dl, feature_name_dl = get_processed_data(
            file_path='data/bank-additional-full.csv',
            test_size=test_size,
            random_state=random_state,
            apply_smote='auto',
            numerical_scaler_type=numerical_scaler_type
        )
    # Ensure input_dim for DL model is correct
    config['deep_learning_model']['input_dim'] =  X_train_dl.shape[1]
    print(f"Deep Learning model input dimension set to: {config['deep_learning_model']['input_dim']}")
    
    print(f"\nData preparation complete.")
    print(f"Classical Models - X_train shape: {X_train_classical}, y_train shape: {y_train_classical.shape}")
    print(f"Deep Learning - X_train shape: {X_train_dl.shape}, y_train shape {y_train_dl.shape}")
    
    
    
    #--- 3. Classical ML Model Training and Evaluation
    print("\n--- Stage 3: Training and Evaluating Classical ML Models ---")
    classical_models_config = config['classical_models']
    
    for model_name, params in classical_models_config.items():
        print(f"\n--- Starting training for {model_name} ---")
        trained_pipeline = train_classical_model(
            model_name = model_name,
            X_train = X_train_classical,
            y_train = y_train_classical,
            X_test = X_test_classical,
            y_test = y_test_classical,
            preprocessor=preprocessor_for_classical,
            model_params=params,
            wandb_project_name=project_name
        )
        print("Finished training and evaluating {model_name}")
    
    #--- Stage 4. Deep Learning Model Training and Evaluation
    print("\n--- Stage 4: Training and Evaluating ----")
    dl_config = config['deep_learning_model']
    
    trained_dl_model = train_deeL_model(
        X_Train = X_train_dl,
        y_train = y_train_dl, 
        X_Test = X_test_dl,
        y_test= y_test_dl,
        dl_config=dl_config,
        wandb_project_name=project_name,
        device=device,
    )
    
    
    print("Finished training and evaluation Deep Learning Model")
    
    #--- 5 Model Comparison and Recommendation
    print("\n --- Stage 5: Model Comparison and Recommendation ---")
    print("All model have been trained and their results logged into Weights & Biases")
    print(f"Please visit you W&B dashboard at  https://wandb.ai/{os.getenv('WANDB_ENTITY')}/{project_name}/runs ")
    print("You can compare models based on metrics (F1-score, ROC AUC are key for imbalanced data), ")
    
    # Ensure any lingering W&B runs are finished
    finish_wandb_run()
    print("\n--- Project Execution complete ---")
    
if __name__ == "__main__":
    main()