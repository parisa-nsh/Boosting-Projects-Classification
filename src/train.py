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
from sklearn.datasets import make_classification

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

def load_data():
    """
    Load and preprocess data
    For this example, we'll create synthetic data
    """
    # Create synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                             n_redundant=5, random_state=42)
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    
    return X, y

def main():
    # Load data
    X, y = load_data()
    
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