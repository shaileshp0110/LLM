from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import numpy as np

class RealisticFraudDetectionModel:
    def __init__(self, params=None):
        """Initialize with tuned parameters for realistic fraud detection"""
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'error'],
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 10,  # Adjusted for class imbalance
            'min_child_weight': 1,
            'random_state': 42
        }
        self.model = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train with early stopping and validation"""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        watchlist = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            watchlist.append((dval, 'eval'))
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,
            early_stopping_rounds=50,
            evals=watchlist,
            verbose_eval=100
        )
        
    def predict(self, X):
        """Make probability predictions"""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate with custom threshold"""
        y_prob = self.predict(X_test)
        y_pred = (y_prob > threshold).astype(int)
        
        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'predictions': y_pred,
            'probabilities': y_prob
        }
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance scores"""
        importance = self.model.get_score(importance_type='gain')
        if feature_names is not None:
            importance = {feature_names[int(k[1:])]: v 
                        for k, v in importance.items()}
        return importance 