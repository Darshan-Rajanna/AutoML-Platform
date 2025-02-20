import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self):
        self.numerical_pipeline = None
        self.categorical_pipeline = None
        self.preprocessor = None
        self.target_encoder = LabelEncoder()
        self.feature_names = None
        self.target_classes = None
        
    def create_preprocessing_pipeline(self, numerical_features, categorical_features):
        """Create preprocessing pipeline for numerical and categorical features."""
        self.numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        self.categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        transformers = []
        if numerical_features:
            transformers.append(('num', self.numerical_pipeline, numerical_features))
        if categorical_features:
            transformers.append(('cat', self.categorical_pipeline, categorical_features))
            
        self.preprocessor = ColumnTransformer(transformers)
        return self.preprocessor
    
    def fit_transform(self, X, y=None, numerical_features=None, categorical_features=None):
        """Fit and transform the data using the preprocessing pipeline."""
        self.feature_names = X.columns.tolist()
        
        if self.preprocessor is None and (numerical_features is not None or categorical_features is not None):
            self.create_preprocessing_pipeline(numerical_features, categorical_features)
            
        if self.preprocessor is None:
            return X
            
        X_transformed = self.preprocessor.fit_transform(X)
        
        # If y is provided, fit the target encoder
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            self.target_encoder.fit(y)
            self.target_classes = self.target_encoder.classes_
            y_transformed = self.target_encoder.transform(y)
            return X_transformed, y_transformed
            
        return X_transformed
    
    def transform(self, X, y=None):
        """Transform new data using the fitted preprocessing pipeline."""
        if self.preprocessor is None:
            return X
            
        X_transformed = self.preprocessor.transform(X)
        
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            y_transformed = self.target_encoder.transform(y)
            return X_transformed, y_transformed
            
        return X_transformed
    
    def inverse_transform_target(self, y):
        """Transform encoded labels back to original classes."""
        if self.target_encoder is None or not hasattr(self.target_encoder, 'classes_'):
            return y
        return self.target_encoder.inverse_transform(y)
    
    @staticmethod
    def identify_feature_types(df):
        """Automatically identify numerical and categorical features."""
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numerical_features, categorical_features
        
    def get_feature_names(self):
        """Get the names of the features after preprocessing."""
        if self.preprocessor is None:
            return self.feature_names
            
        feature_names = []
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                encoder = transformer.named_steps['onehot']
                for feature in features:
                    feature_names.extend([f"{feature}_{val}" for val in encoder.get_feature_names_out([feature])])
        return feature_names
