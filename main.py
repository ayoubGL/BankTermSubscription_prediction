import os
import yaml
import pandas as pd
import numpy as np


from src.data_pipeline import get_processed_data
from src.classical_models import train_classical_model

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
    
    # --- 2. Data Pipelinee
    print("\n--- Stage 2: Running Data Pipeline ---")
    X_train_raw_df, X_test_raw_df, y_train, y_test, full_pipeline, feature_name_after_preprocessing = get_processed_data(
        file_path = "data/bank-additional-full.csv",
        test_size=test_size,
        random_state=random_state,
        apply_smote=apply_smote_classical,
        numerical_scaler_type=numerical_scaler_type
    )
    
    preprocessor_for_classical = full_pipeline.named_steps['preprocessor']
    
    print(f"\nData preparation complete. X_train_raw_df shape: {X_train_raw_df.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_raw_df shape: {X_test_raw_df.shape}, y_test shape: {y_test.shape}")
    print(f"Number of features after preprocessing: {len(feature_name_after_preprocessing)}")
    
    
    #--- 3. Classical ML Model Training and Evaluation
    print("\n--- Stage 3: Training and Evaluating Classical ML Models ---")
    classical_models_config = config['classical_models']
    
    for model_name, params in classical_models_config.items():
        print(f"\n--- Starting training for {model_name} ---")
        trained_pipeline = train_classical_model(
            model_name = model_name,
            X_train = X_train_raw_df,
            y_train = y_train,
            X_test = X_test_raw_df,
            y_test = y_test,
            preprocessor=preprocessor_for_classical,
            model_params=params,
            wandb_project_name=project_name
        )
        print("Finished training and evaluating {model_name}")
    
if __name__ == "__main__":
    main()