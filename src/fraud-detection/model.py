import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

class FraudDetectionModel:
    def __init__(self, params=None):
        """
        Initialize the XGBoost fraud detection model
        
        Parameters:
        params (dict): XGBoost parameters. If None, uses default parameters
        """
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,  # Adjust based on class imbalance
            'random_state': 42
        }
        self.model = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None, num_rounds=100):
        """
        Train the XGBoost model
        
        Parameters:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        num_rounds: Number of boosting rounds
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Create validation set if provided
        watchlist = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            watchlist.append((dval, 'eval'))
        
        # Train the model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_rounds,
            watchlist,
            early_stopping_rounds=20,
            verbose_eval=10
        )
        
    def predict(self, X):
        """Make probability predictions"""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Evaluate the model performance
        
        Parameters:
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold for converting probabilities to labels
        """
        # Get probability predictions
        y_prob = self.predict(X_test)
        y_pred = (y_prob > threshold).astype(int)
        
        # Calculate metrics
        results = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        return results
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance scores"""
        importance = self.model.get_score(importance_type='gain')
        if feature_names is not None:
            importance = {feature_names[int(k[1:])]: v 
                        for k, v in importance.items()}
        return importance 