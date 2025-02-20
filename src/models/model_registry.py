from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

class ModelRegistry:
    @staticmethod
    def get_classification_models():
        """Return a dictionary of classification models with their default parameters."""
        return {
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': (0.01, 100),
                    'max_iter': (100, 500)
                }
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': (50, 300),
                    'max_depth': (3, 15),
                    'min_samples_split': (2, 20)
                }
            },
            'svm': {
                'model': SVC,
                'params': {
                    'C': (0.01, 100),
                    'gamma': (0.001, 1.0)
                }
            },
            'knn': {
                'model': KNeighborsClassifier,
                'params': {
                    'n_neighbors': (3, 15),
                    'leaf_size': (20, 50)
                }
            },
            'lightgbm': {
                'model': LGBMClassifier,
                'params': {
                    'num_leaves': (20, 100),
                    'learning_rate': (0.01, 0.3),
                    'n_estimators': (50, 300)
                }
            },
            'xgboost': {
                'model': XGBClassifier,
                'params': {
                    'max_depth': (3, 15),
                    'learning_rate': (0.01, 0.3),
                    'n_estimators': (50, 300)
                }
            }
        }

    @staticmethod
    def get_regression_models():
        """Return a dictionary of regression models with their default parameters."""
        return {
            'linear_regression': {
                'model': LinearRegression,
                'params': {}
            },
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': (50, 300),
                    'max_depth': (3, 15),
                    'min_samples_split': (2, 20)
                }
            },
            'svr': {
                'model': SVR,
                'params': {
                    'C': (0.01, 100),
                    'gamma': (0.001, 1.0)
                }
            },
            'lightgbm': {
                'model': LGBMRegressor,
                'params': {
                    'num_leaves': (20, 100),
                    'learning_rate': (0.01, 0.3),
                    'n_estimators': (50, 300)
                }
            },
            'xgboost': {
                'model': XGBRegressor,
                'params': {
                    'max_depth': (3, 15),
                    'learning_rate': (0.01, 0.3),
                    'n_estimators': (50, 300)
                }
            }
        }
