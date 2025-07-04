import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE # For handling class imbalance
from imblearn.pipeline import Pipeline as ImbPipline


def load_data(file_path: str = "../data/bank-additional-full.csv"):
    """
        Loads the raw Bank Marketing dataset.
        
        Args:
            file_path (str): Path to the CSV file.
        
        Returns:
            pd.DataFrame: Loaded DataFrame
    """
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep=';')
    print("Data loaded successfully.")
    return df

def clean_data(df: pd.DataFrame):
    """
        Performs initial data cleaning steps based on EDA insights
        
        Args:
            df: Input DataFrame
            
        Returns
            pd.DataFrame: cleaned DF
    """
    
    print("Preforming initial data cleaning...")

    # Convert target variable 'y' to numerical
    df['y'] = df['y'].map({'no':0, 'yes':1})
    print("Converted target 'y' to numerical (0/1).")
    
    # Handle 'pdays' special value (999)
    df['pdays_not_contacted'] = (df['pdays'] == 999).astype(int) 
    print('Created pdays_not_contacted for pdays === 999.')
    
    # Handle 'unknown'  values in categorical columns
    # No explicit action needed here, OneHotEncoder will handle it as a separate category.
    print("Treating 'unknown' in categorical columns as distinct categories")
    print("Initial data cleaning complete.")
    
    return df
    
    
def create_preprocessing_pipeline(df: pd.DataFrame, numerical_scaler_type: str = 'standard'):
        """
        Creates a scikit-learn preprocessing pipeline using ColumnTransformer
        
        Args:
            df: The DataFrame to infer column type from.
            numerical_scaler_type: Type of scaler
        
        Returns:
            ColumnTransformer: A preprocessor object ready for fitting and transforming data
        """
        
        print("Creating preprocessing pipeline ...")
        
        # Identify numerical and categorical
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        
        # Ensure 'y' is not in features
        if 'y' in numerical_cols:
            numerical_cols.remove('y')
        if 'y' in categorical_cols:
            categorical_cols.remove('y')
            
        # Ensure 'pydays_not_contacted' is in numerical_cols
        if 'pdays_not_contacted' in categorical_cols:
            categorical_cols.remove('pdays_not_contacted')
        if 'pdays_not__contacted' not in numerical_cols:
            numerical_cols.remove('pdays_not_contacted')
        
        
        print(f" Identified numerical columns: {numerical_cols}")
        print(f" Identified categorical columns: {categorical_cols}")
        
        # Define numerical transformer
        if numerical_scaler_type == "standard":
            numerical_transformer = StandardScaler()
            print("Using StandardScaler for numerical features.")
        elif numerical_scaler_type == 'minmax':
            numerical_transformer = MinMaxScaler()
            print("Using MinMaxScaler for numerical features")
        else:
            raise ValueError("numerical_scaler_type must be 'standard' or 'minmax' ")
        
        
        # Define categorical transformer (One-Hot Encoding)
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        print("Using OneHotEncoder for categorical  features.")
        
        # Create columnTransformer, and use reminder, to keep all the columns that doesn't need to be transformed
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'
        )
        print("Preprocessing pipeline created.")
        return preprocessor, numerical_cols, categorical_cols
    
    
def get_processed_data(file_path: str = '../data/bank-additional-full.csv',
                           test_size: float = 0.2, random_state: int = 42,
                           apply_smote: bool = False, smote_sampling_strategy: str = 'auto',
                           numerical_scaler_type :str = 'standard'
                           ):
        
        """
        Loads, cleans, preprocess, and split the data, optionally applying SMOTE.
        
        Args:
            file_path (str): Path of the raw csv data
            test_size (float): Proportion of the dataset to include in the test split
            random_state(int): Controls the shuffling applied to the data before spliting
            apply_smote (bool): Whether to apply SMOTE to the training 
            smote_sampling_strategy(str): Sampling strategy for SMOTE
            numerical_scaler_type (str): Type of scaler for numerical features('standard' or minmax).
        Return:
            tuple: X_train, X_test, y_train, y_test, preprocessor, features_names_after_preprocessing
        """
        
        print("\n -- Starting full data pipeline --")
        df = load_data(file_path)
        df = clean_data(df)
        
        # Separate features (X) and target (y)
        X = df.drop('y', axis=1)
        y = df['y']
        print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
        
        # Split the data into training and testing
        # Stratify by 'y' to maintain the same class distribution in train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Data split into training ({X_train.shape[0]} sample) and test ({X_test.shape[0]} samples).")    
        
        preprocessor, numerical_cols, categorical_cols = create_preprocessing_pipeline(
            X_train, numerical_scaler_type)
        
        # If applying SMOTE
        if apply_smote:
            # For SMOTE it's crucial to apply it ONLY on the training data AFTER splitting
            # and AFTEr numerical features are scales, but BEFORE one-hot encoding if possible 
            # for better performance
            # imblearn pipeline handles this correctly
            print(f"Applying SMOTE with sampling strategy:'{smote_sampling_strategy}'")
            # The pipeline will first preprocess, then apply SMOTE
            full_pipeline = ImbPipline(steps = [
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state = random_state, sampling_strategy=smote_sampling_strategy))
            ])
            
            # Fit and transform training data
            X_train_processed, y_train_preprocessed = full_pipeline.fit_resample(X_train, y_train)
            
            # Transform data using only the preprocessor
            X_test_processed = preprocessor.transform(X_test)
            print(f"SMOTE applied. New training data shape: {X_train_processed.shape}")
            print(f"New training target distribution:\n{pd.Series(y_train_preprocessed).value_counts(normalize=True)}")
        else:
            print("SMOTE not applied")
            full_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
            # Fit and transform training data
            X_train_processed = full_pipeline.fit_transform(X_train, y_train)
            # Transform test data
            X_test_processed = full_pipeline.transform(X_test)
            
            
        
        # Get feature names after preprocessing (important for interpretability and deep learning input)    
        # For OneHotEncoder, get_feature_names_out() is available after fit.
        # For numerical features, their names remain the same.
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
        feature_names_after_preprocessing = numerical_cols + ohe_feature_names
        print(f"Total features after preprocessing: {len(feature_names_after_preprocessing)}")
        
        print("Full data pipeline complete")
        return X_train_processed, X_test_processed, y_train_preprocessed, y_test, full_pipeline, feature_names_after_preprocessing

if __name__ == "__main__":
    # Example usage:
    X_train, X_test, y_train, y_test, pipeline, feature_names = get_processed_data(apply_smote=True)
    
    print("\n--- Sample of Processed Data ---")
    print(f"X_train_processed shape: {X_train.shape}")
    print(f"y_train_processed shape: {y_train.shape}")
    print(f"X_test_processed shape: {X_test.shape}")
    print(f"y_test_processed shape: {y_test.shape}")

    # Display a small part of the processed training data
    # Note: X_train is a numpy array after preprocessing
    print("\nFirst 5 rows of X_train_processed (numerical representation):")
    print(X_train[:5, :5]) # Print first 5 rows and first 5 columns

    print("\nFirst 5 rows of y_train_processed:")
    print(y_train[:5])

    print("\nFirst 5 feature names after preprocessing:")
    print(feature_names[:5])

    # You can also inspect the pipeline steps
    print("\nPipeline steps:")
    for step_name, step_transformer in pipeline.steps:
        print(f"- {step_name}: {step_transformer}")

