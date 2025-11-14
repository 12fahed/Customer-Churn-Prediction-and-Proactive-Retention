"""
Script to pre-train models with optimal hyperparameters
Run this once to create pre-trained models for the CRM system
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Create models directory
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_and_preprocess_data():
    """Load and preprocess the training data"""
    print("Loading data...")
    df_train = pd.read_csv('data/train.csv')
    
    # Handle missing values
    for col in df_train.columns:
        if df_train[col].dtype == 'object':
            df_train[col].fillna(df_train[col].mode()[0] if not df_train[col].mode().empty else 'Unknown', inplace=True)
        else:
            df_train[col].fillna(df_train[col].median(), inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df_train.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col != 'Customer ID':
            df_train[f'{col}_encoded'] = le.fit_transform(df_train[col].astype(str))
    
    # Feature engineering
    print("Engineering features...")
    
    # Total services
    service_cols = ['Phone Service', 'Multiple Lines', 'Internet Service', 
                   'Online Security', 'Online Backup', 'Device Protection Plan',
                   'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music']
    
    for col in service_cols:
        if col in df_train.columns:
            df_train[col + '_binary'] = (df_train[col] == 'Yes').astype(int)
    
    service_binary_cols = [col + '_binary' for col in service_cols if col in df_train.columns]
    if service_binary_cols:
        df_train['TotalServices'] = df_train[service_binary_cols].sum(axis=1)
    
    # High spender flag
    if 'Monthly Charge' in df_train.columns:
        df_train['HighSpender'] = (df_train['Monthly Charge'] > df_train['Monthly Charge'].quantile(0.75)).astype(int)
    
    # Tenure groups
    if 'Tenure in Months' in df_train.columns:
        df_train['TenureGroup'] = pd.cut(df_train['Tenure in Months'], 
                                        bins=[0, 12, 24, 48, 100], 
                                        labels=['0-1Y', '1-2Y', '2-4Y', '4Y+'])
    
    print(f"Data shape after preprocessing: {df_train.shape}")
    return df_train

def select_features(df):
    """Select relevant features for modeling"""
    # Recommended features
    recommended_features = [
        'Tenure in Months', 'Monthly Charge', 'Total Charges', 'Total Revenue',
        'Age', 'Number of Dependents', 'Number of Referrals', 'Satisfaction Score',
        'CLTV', 'Churn Score', 'TotalServices', 'HighSpender'
    ]
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove non-predictive columns
    non_predictive = ['Customer ID', 'Zip Code', 'Latitude', 'Longitude', 'Population', 'Churn']
    numeric_cols = [col for col in numeric_cols if col not in non_predictive]
    
    # Get categorical encoded columns
    categorical_encoded = [col for col in df.columns if col.endswith('_encoded')]
    
    # Combine and filter
    all_features = numeric_cols + categorical_encoded
    selected = [col for col in recommended_features if col in all_features]
    
    # If recommended features not enough, add more
    if len(selected) < 10:
        additional = [col for col in all_features if col not in selected][:15]
        selected.extend(additional)
    
    print(f"Selected {len(selected)} features for modeling")
    return selected

def train_optimized_models(X_train, y_train, X_test, y_test):
    """Train models with optimized hyperparameters"""
    print("\nTraining optimized models...")
    
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=1.0,
            solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Training metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Test metrics
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results[name] = {
            'model': model,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'test_precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'test_f1': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
            'train_roc_auc': roc_auc_score(y_train, y_train_proba) if y_train_proba is not None else None,
            'test_roc_auc': roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else None
        }
        
        print(f"{name} - Test Accuracy: {results[name]['test_accuracy']:.4f}, Test F1: {results[name]['test_f1']:.4f}")
    
    return results

def save_trained_models(models_dict, feature_names):
    """Save models and metadata"""
    print("\nSaving models...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_paths = {}
    for model_name, model_data in models_dict.items():
        filename = f"{model_name.replace(' ', '_').lower()}_{timestamp}.pkl"
        filepath = os.path.join(MODEL_DIR, filename)
        joblib.dump(model_data['model'], filepath)
        model_paths[model_name] = filepath
        print(f"Saved {model_name} to {filepath}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'feature_names': feature_names,
        'model_paths': model_paths,
        'model_info': {
            'training_date': datetime.now().isoformat(),
            'features_count': len(feature_names)
        },
        'performance_metrics': {
            name: {
                'accuracy': data['test_accuracy'],
                'precision': data['test_precision'],
                'recall': data['test_recall'],
                'f1_score': data['test_f1'],
                'roc_auc': data.get('test_roc_auc'),
                'test_accuracy': data['test_accuracy'],
                'test_f1_score': data['test_f1']
            }
            for name, data in models_dict.items()
        }
    }
    
    metadata_path = os.path.join(MODEL_DIR, f'metadata_{timestamp}.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"Saved metadata to {metadata_path}")
    
    return timestamp

def main():
    print("=" * 60)
    print("GRAHAK CRM - Pre-training Models")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Select features
    features = select_features(df)
    
    # Prepare data
    X = df[features].copy()
    y = df['Churn'].copy()
    
    # Clean data
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Churn rate (train): {y_train.mean():.2%}")
    print(f"Churn rate (test): {y_test.mean():.2%}")
    
    # Train models
    results = train_optimized_models(X_train, y_train, X_test, y_test)
    
    # Save models
    timestamp = save_trained_models(results, features)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Models saved with timestamp: {timestamp}")
    print("=" * 60)
    
    # Display summary
    print("\nModel Performance Summary:")
    print("-" * 60)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Test Accuracy:  {result['test_accuracy']:.4f}")
        print(f"  Test Precision: {result['test_precision']:.4f}")
        print(f"  Test Recall:    {result['test_recall']:.4f}")
        print(f"  Test F1-Score:  {result['test_f1']:.4f}")
        if result['test_roc_auc']:
            print(f"  Test ROC-AUC:   {result['test_roc_auc']:.4f}")
    
    # Best model
    best_model = max(results.items(), key=lambda x: x[1]['test_f1'])
    print("\n" + "=" * 60)
    print(f"Best Model: {best_model[0]}")
    print(f"F1-Score: {best_model[1]['test_f1']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
