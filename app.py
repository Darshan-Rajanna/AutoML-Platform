from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from src.preprocessing.preprocessor import DataPreprocessor
from src.models.model_registry import ModelRegistry
from src.optimization.bayesian_optimizer import BayesianOptimizer
from src.evaluation.metrics import ModelEvaluator
import joblib
import json
import scipy
import os

app = Flask(__name__)

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        df = pd.read_csv(file)
        return jsonify({
            'message': 'File uploaded successfully',
            'columns': df.columns.tolist(),
            'sample': df.head().to_dict('records'),  # First few rows as records
            'data': df.to_dict('records')  # Full dataset as records
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Get data and parameters from request
        data = request.json
        
        # Debug: Print received data structure
        print("Received data structure:", data.keys())
        
        df = pd.DataFrame(data['data'])
        target_column = data['target_column']
        task_type = data['task_type']  # 'classification' or 'regression'
        
        # Debug: Print data info
        print("\nDataFrame Info:")
        print(df.info())
        print("\nTarget Column:", target_column)
        print("Task Type:", task_type)
        print("\nUnique values in target column:", df[target_column].unique())
        print("Value counts in target column:\n", df[target_column].value_counts())
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Debug: Print shape of X and y
        print("\nShape of X:", X.shape)
        print("Shape of y:", y.shape)
        print("Sample of X:\n", X.head())
        print("Sample of y:\n", y.head())
        
        # Check class balance for classification tasks
        if task_type == 'classification':
            class_counts = y.value_counts()
            print("\nClass distribution:", class_counts)
            
            # Check if the target column is empty or contains only null values
            if y.isna().all():
                return jsonify({
                    'error': 'Target column contains only null values.'
                }), 400
                
            # Check if we have at least 2 unique non-null classes
            unique_classes = y.dropna().unique()
            if len(unique_classes) < 2:
                return jsonify({
                    'error': f'Data must contain at least 2 classes for classification tasks. Currently found classes: {unique_classes}'
                }), 400
            
            # Check for severe class imbalance
            min_class_count = class_counts.min()
            max_class_count = class_counts.max()
            imbalance_ratio = min_class_count / max_class_count
            print(f"\nClass imbalance ratio: {imbalance_ratio}")
            
            if imbalance_ratio < 0.1:  # Less than 10% ratio
                return jsonify({
                    'error': f'Severe class imbalance detected. Minimum class has {min_class_count} samples, maximum class has {max_class_count} samples.'
                }), 400
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        numerical_features, categorical_features = preprocessor.identify_feature_types(X)
        
        print("\nFeature types:")
        print("Numerical features:", numerical_features)
        print("Categorical features:", categorical_features)
        
        # Process features and target
        if task_type == 'classification':
            X_processed, y_processed = preprocessor.fit_transform(X, y, numerical_features, categorical_features)
            # Store target classes for later use
            target_classes = preprocessor.target_classes.tolist()
            print("\nEncoded target classes:", target_classes)
            print("Unique values in processed target:", np.unique(y_processed))
        else:
            X_processed = preprocessor.fit_transform(X, numerical_features=numerical_features, categorical_features=categorical_features)
            y_processed = y
            target_classes = None
        
        # Convert sparse matrix to dense if necessary
        if scipy.sparse.issparse(X_processed):
            X_processed = X_processed.toarray()
            
        print("\nProcessed data shapes:")
        print("X_processed shape:", X_processed.shape)
        print("y_processed shape:", y_processed.shape if isinstance(y_processed, np.ndarray) else len(y_processed))
        
        # Get models based on task type
        if task_type == 'classification':
            models = ModelRegistry.get_classification_models()
            scoring = 'accuracy'
        else:
            models = ModelRegistry.get_regression_models()
            scoring = 'neg_mean_squared_error'
            
        # Optimize each model
        results = {}
        for model_name, model_info in models.items():
            print(f"\nTraining model: {model_name}")
            optimizer = BayesianOptimizer(
                model_info['model'],
                model_info['params'],
                scoring=scoring,
                direction='maximize' if task_type == 'classification' else 'minimize',
                task_type=task_type
            )
            
            try:
                optimization_result = optimizer.optimize(X_processed, y_processed)
                results[model_name] = {
                    'best_params': optimization_result['best_params'],
                    'best_score': float(optimization_result['best_score']),
                    'optimization_history': optimizer.get_optimization_history()
                }
                print(f"Successfully trained {model_name}")
                print(f"Best score: {optimization_result['best_score']}")
                
                # Save the best model and preprocessor
                model_path = os.path.join(MODELS_DIR, f'{model_name}_best.joblib')
                model_artifacts = {
                    'model': optimization_result['best_model'],
                    'preprocessor': preprocessor,
                    'target_classes': target_classes,
                    'feature_names': preprocessor.get_feature_names()
                }
                joblib.dump(model_artifacts, model_path)
                print(f"Saved model artifacts to: {model_path}")
                
            except Exception as e:
                print(f"Failed to optimize {model_name}: {str(e)}")
                continue
            
        if not results:
            return jsonify({
                'error': 'All models failed to train. Please check your data and try again.'
            }), 400
            
        return jsonify({
            'message': 'Training completed successfully',
            'results': results,
            'target_classes': target_classes if task_type == 'classification' else None
        })
        
    except Exception as e:
        import traceback
        print("Exception details:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_name = data['model_name']
        input_data = pd.DataFrame(data['data'])
        
        # Load model and preprocessor
        model_path = os.path.join(MODELS_DIR, f'{model_name}_best.joblib')
        model_artifacts = joblib.load(model_path)
        model = model_artifacts['model']
        preprocessor = model_artifacts['preprocessor']
        
        # Preprocess input data
        X_processed = preprocessor.transform(input_data)
        
        # Make predictions
        predictions = model.predict(X_processed)
        
        # If classification, convert predictions back to original classes
        if 'target_classes' in model_artifacts and model_artifacts['target_classes'] is not None:
            predictions = preprocessor.inverse_transform_target(predictions)
        
        return jsonify({
            'predictions': predictions.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/download_model/<model_name>', methods=['GET'])
def download_model(model_name):
    try:
        # Construct the model file path
        model_path = os.path.join(MODELS_DIR, f'{model_name}_best.joblib')
        
        # Check if model exists
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model {model_name} not found'}), 404
            
        # Send the file
        return send_file(
            model_path,
            as_attachment=True,
            download_name=f'{model_name.lower()}_model.pkl',
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
