import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import (load_breast_cancer, load_iris, 
                            fetch_california_housing, load_wine)

class BoostingClassifier:
    def __init__(self, model_type='xgboost', params=None):
        """
        Initialize the boosting classifier
        Args:
            model_type (str): Type of boosting algorithm ('xgboost', 'lightgbm', 'catboost', 'gradientboost')
            params (dict): Parameters for the model
        """
        self.model_type = model_type.lower()
        self.params = params if params is not None else {}
        self.model = self._get_model()
        
    def _get_model(self):
        """
        Get the appropriate model based on model_type
        """
        if self.model_type == 'xgboost':
            return xgb.XGBClassifier(**self.params)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(**self.params)
        elif self.model_type == 'catboost':
            return CatBoostClassifier(**self.params)
        elif self.model_type == 'gradientboost':
            return GradientBoostingClassifier(**self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        """
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)],
                         verbose=True)
        else:
            self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        return accuracy, report
    
    def plot_feature_importance(self, feature_names):
        """
        Plot feature importance
        """
        plt.figure(figsize=(10, 6))
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            return
        
        indices = np.argsort(importances)[::-1]
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

def load_data(dataset_name='breast_cancer'):
    """
    Load and preprocess data from various available datasets
    Args:
        dataset_name (str): Name of the dataset to load
            Options: 'breast_cancer', 'iris', 'wine', 'california'
    Returns:
        X (pd.DataFrame): Features
        y (np.array): Target variable
    """
    datasets = {
        'breast_cancer': load_breast_cancer,
        'iris': load_iris,
        'wine': load_wine,
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(datasets.keys())}")
    
    # Load the selected dataset
    data = datasets[dataset_name]()
    
    # Convert to DataFrame with feature names
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    print(f"\nDataset: {data.DESCR.split('Description')[0]}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target classes: {np.unique(y)}")
    print(f"Features: {', '.join(data.feature_names)}\n")
    
    return X, y

def main():
    # Load data - you can change the dataset here
    X, y = load_data('breast_cancer')  # or 'iris', 'wine'
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to try
    models = {
        'xgboost': {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100},
        'lightgbm': {'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 100},
        'catboost': {'depth': 3, 'learning_rate': 0.1, 'iterations': 100},
        'gradientboost': {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
    }
    
    # Train and evaluate each model
    results = {}
    for model_name, params in models.items():
        print(f"\nTraining {model_name}...")
        classifier = BoostingClassifier(model_type=model_name, params=params)
        classifier.train(X_train_scaled, y_train)
        
        # Evaluate
        accuracy, report = classifier.evaluate(X_test_scaled, y_test)
        results[model_name] = accuracy
        
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        
        # Plot feature importance
        classifier.plot_feature_importance(X.columns)
    
    # Plot comparison of models
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Model Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

if __name__ == "__main__":
    main() 