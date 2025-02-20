import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import numpy as np
from sklearn.exceptions import NotFittedError

class BayesianOptimizer:
    def __init__(self, model_class, param_space, n_trials=100, cv=5, scoring='accuracy', direction='maximize', task_type='classification'):
        self.model_class = model_class
        self.param_space = param_space
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.direction = direction
        self.task_type = task_type
        self.study = None
        self.best_model = None
        
    def objective(self, trial, X, y):
        """Objective function for Optuna optimization."""
        params = {}
        for param_name, param_range in self.param_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], log=True)
        
        model = self.model_class(**params)
        
        try:
            # Use StratifiedKFold for classification tasks to maintain class distribution
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=self.cv, shuffle=True, random_state=42)
                
            scores = cross_val_score(
                model, X, y, 
                cv=cv, 
                scoring=self.scoring,
                error_score='raise'
            )
            return scores.mean()
        except Exception as e:
            print(f"Trial failed with error: {str(e)}")
            # Return a very poor score to indicate failure
            return float('-inf') if self.direction == 'maximize' else float('inf')
    
    def optimize(self, X, y):
        """Run Bayesian optimization to find the best hyperparameters."""
        study = optuna.create_study(direction=self.direction)
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=self.n_trials)
        
        self.study = study
        
        # Train the best model
        best_params = study.best_params
        self.best_model = self.model_class(**best_params)
        
        try:
            self.best_model.fit(X, y)
        except Exception as e:
            raise ValueError(f"Failed to train the best model: {str(e)}")
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'best_model': self.best_model
        }
    
    def get_optimization_history(self):
        """Return the optimization history for visualization."""
        if self.study is None:
            raise ValueError("No optimization has been performed yet.")
            
        return {
            'values': self.study.trials_dataframe()['value'].tolist(),
            'params': [t.params for t in self.study.trials]
        }
