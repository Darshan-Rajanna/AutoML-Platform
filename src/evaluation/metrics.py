from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np

class ModelEvaluator:
    @staticmethod
    def evaluate_classification(y_true, y_pred, y_prob=None):
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for ROC-AUC)
            
        Returns:
            Dictionary containing various classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except:
                metrics['roc_auc'] = None
                
        return metrics
    
    @staticmethod
    def evaluate_regression(y_true, y_pred):
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing various regression metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    @staticmethod
    def evaluate_time_series(y_true, y_pred):
        """
        Evaluate time series model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing various time series metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'mape': mape
        }
