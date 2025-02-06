from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import numpy as np

class SimpleFraudDetectionModel:
    def __init__(self, params=None):
        """Initialize with simple parameters for fraud detection"""
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'random_state': 42
        }
        self.model = None
        
    def train(self, X_train, y_train):
        """Train the simple fraud detection model"""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(
            self.params, 
            dtrain, 
            num_boost_round=100
        )
        
    def predict(self, X):
        """Make probability predictions"""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_prob = self.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int)
        
        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'predictions': y_pred,
            'probabilities': y_prob
        } 