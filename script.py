import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import joblib
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.impute import SimpleImputer

# Association Rules
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# NLP Libraries
import re
from textblob import TextBlob
import random

# AI Algorithm (Genetic Algorithm)
import random as rand
from deap import base, creator, tools, algorithms

# Set page config
st.set_page_config(
    page_title="GRAHAK - Intelligent CRM System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'clusters_created' not in st.session_state:
    st.session_state.clusters_created = False
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'target_column' not in st.session_state:
    st.session_state.target_column = 'Churn'
if 'test_data_loaded' not in st.session_state:
    st.session_state.test_data_loaded = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'model_info' not in st.session_state:
    st.session_state.model_info = {}

# Model storage directory
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Data cleaning and validation functions
def clean_corrupted_data(df):
    """Clean and fix corrupted data in the dataset."""
    df_clean = df.copy()
    # st.info(" Cleaning data...")
    
    # Handle missing values in key columns
    # For Churn Category and Churn Reason, nulls mean no churn (they only exist for churned customers)
    if 'Churn Category' in df_clean.columns:
        df_clean['Churn Category'] = df_clean['Churn Category'].fillna('No Churn')
    
    if 'Churn Reason' in df_clean.columns:
        df_clean['Churn Reason'] = df_clean['Churn Reason'].fillna('No Churn')
    
    # Handle Internet Type nulls
    if 'Internet Type' in df_clean.columns:
        df_clean['Internet Type'] = df_clean['Internet Type'].fillna('No Internet')
    
    # Handle Offer nulls
    if 'Offer' in df_clean.columns:
        df_clean['Offer'] = df_clean['Offer'].fillna('No Offer')
    
    return df_clean

# Data Loading Functions
def load_real_data():
    """Load the real datasets from the data folder."""
    try:
        data_path = "data/"
        
        # Check if data files exist
        train_path = os.path.join(data_path, "train.csv")
        test_path = os.path.join(data_path, "test.csv")
        
        datasets = {}
        
        if os.path.exists(train_path):
            df_train = pd.read_csv(train_path)
            
            # Clean any corrupted data
            df_train = clean_corrupted_data(df_train)
            datasets['train'] = df_train
            
            if os.path.exists(test_path):
                df_test = pd.read_csv(test_path)
                
                # Clean test data
                df_test = clean_corrupted_data(df_test)
                datasets['test'] = df_test
                st.session_state.test_data_loaded = True
            
            return df_train, datasets
        else:
            return None, None

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def preprocess_real_data(df):
    """Preprocess the real dataset for analysis."""
    df_processed = df.copy()
    
    # Handle missing values
    if df_processed.isnull().sum().sum() > 0:
        st.warning("Missing values detected. Handling missing data...")
        
        # Fill numerical missing values with median
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                mode_value = df_processed[col].mode()
                if len(mode_value) > 0:
                    df_processed[col].fillna(mode_value[0], inplace=True)
                else:
                    df_processed[col].fillna('Unknown', inplace=True)
    
    # Encode categorical variables
    le_dict = {}
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col not in ['Customer ID', 'Lat Long']:  # Skip ID and coordinate columns
            try:
                le = LabelEncoder()
                # Convert any remaining NaN to string
                clean_values = df_processed[col].fillna('Unknown')
                df_processed[col + '_encoded'] = le.fit_transform(clean_values)
                le_dict[col] = le
            except Exception as e:
                st.warning(f"Could not encode column '{col}': {str(e)}")
                continue
    
    st.session_state.label_encoders = le_dict
    
    return df_processed

def create_engineered_features_real(df):
    """Create engineered features based on the actual dataset structure."""
    df_eng = df.copy()
    
    # Check which columns exist and create features accordingly
    available_cols = df.columns.tolist()
    
    # Service count (count binary service columns)
    binary_service_cols = [
        'Phone Service', 'Multiple Lines', 'Internet Service', 
        'Online Security', 'Online Backup', 'Device Protection Plan',
        'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 
        'Streaming Music', 'Unlimited Data'
    ]
    
    # Only include columns that actually exist in the dataset
    existing_binary_cols = [col for col in binary_service_cols if col in df_eng.columns]
    if len(existing_binary_cols) > 0:
        df_eng['TotalServices'] = df_eng[existing_binary_cols].sum(axis=1)
    
    # Tenure grouping
    if 'Tenure in Months' in df_eng.columns:
        df_eng['TenureGroup'] = pd.cut(df_eng['Tenure in Months'], 
                                      bins=[0, 12, 24, 48, float('inf')], 
                                      labels=['New (0-12)', 'Established (13-24)', 'Veteran (25-48)', 'Long-term (49+)'])
    
    # Monthly charges analysis
    if 'Monthly Charge' in df_eng.columns:
        df_eng['HighSpender'] = (df_eng['Monthly Charge'] > df_eng['Monthly Charge'].quantile(0.75)).astype(int)
    
    # Contract type analysis
    if 'Contract' in df_eng.columns:
        df_eng['MonthToMonth'] = (df_eng['Contract'] == 'Month-to-Month').astype(int)
        df_eng['OneYear'] = (df_eng['Contract'] == 'One Year').astype(int)
        df_eng['TwoYear'] = (df_eng['Contract'] == 'Two Year').astype(int)
    
    # Payment method analysis
    if 'Payment Method' in df_eng.columns:
        df_eng['BankWithdrawal'] = (df_eng['Payment Method'] == 'Bank Withdrawal').astype(int)
        df_eng['CreditCard'] = (df_eng['Payment Method'] == 'Credit Card').astype(int)
        df_eng['MailedCheck'] = (df_eng['Payment Method'] == 'Mailed Check').astype(int)
    
    # Age-based features
    if 'Age' in df_eng.columns:
        df_eng['SeniorCitizen'] = (df_eng['Age'] >= 65).astype(int)
        df_eng['YoungCustomer'] = (df_eng['Age'] < 30).astype(int)
    
    return df_eng

def simulate_nlp_features_real(df):
    """Add simulated NLP features to complement real data."""
    np.random.seed(42)
    n_samples = len(df)
    
    # Simulate customer support interaction data
    df['SupportTickets_Last90Days'] = np.random.poisson(2, n_samples)
    df['BillingTickets_Last90Days'] = np.random.poisson(0.5, n_samples)
    df['DaysSinceLastTicket'] = np.random.exponential(30, n_samples).astype(int)
    
    # Simulate sentiment analysis of last ticket
    sentiments = np.random.choice(['Positive', 'Neutral', 'Negative'], 
                                 n_samples, p=[0.3, 0.4, 0.3])
    df['LastTicketSentiment'] = sentiments
    
    # Simulate competitor mentions
    df['MentionedCompetitor'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    return df

# Model Persistence Functions
def save_models(models_dict, feature_names, model_info):
    """Save trained models and metadata to disk."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save each model
        model_paths = {}
        for model_name, model_data in models_dict.items():
            model_filename = f"{model_name.replace(' ', '_').lower()}_{timestamp}.pkl"
            model_path = os.path.join(MODEL_DIR, model_filename)
            joblib.dump(model_data['model'], model_path)
            model_paths[model_name] = model_path
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'feature_names': feature_names,
            'model_paths': model_paths,
            'model_info': model_info,
            'performance_metrics': {
                name: {
                    'accuracy': data['accuracy'],
                    'precision': data['precision'],
                    'recall': data['recall'],
                    'f1_score': data['f1_score'],
                    'roc_auc': data.get('roc_auc'),
                    'test_accuracy': data.get('test_accuracy'),
                    'test_f1_score': data.get('test_f1_score')
                }
                for name, data in models_dict.items()
            }
        }
        
        metadata_path = os.path.join(MODEL_DIR, f'metadata_{timestamp}.pkl')
        joblib.dump(metadata, metadata_path)
        
        return True, timestamp
    except Exception as e:
        st.error(f"Error saving models: {str(e)}")
        return False, None

def load_latest_models():
    """Load the most recent trained models from disk."""
    try:
        if not os.path.exists(MODEL_DIR):
            return None, None, None
        
        # Find latest metadata file
        metadata_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('metadata_')]
        if not metadata_files:
            return None, None, None
        
        latest_metadata = sorted(metadata_files)[-1]
        metadata_path = os.path.join(MODEL_DIR, latest_metadata)
        metadata = joblib.load(metadata_path)
        
        # Load models
        models = {}
        for model_name, model_path in metadata['model_paths'].items():
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
        
        return models, metadata['feature_names'], metadata
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def get_available_model_versions():
    """Get list of all available model versions."""
    try:
        if not os.path.exists(MODEL_DIR):
            return []
        
        metadata_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('metadata_')]
        versions = []
        
        for metadata_file in sorted(metadata_files, reverse=True):
            metadata_path = os.path.join(MODEL_DIR, metadata_file)
            metadata = joblib.load(metadata_path)
            versions.append({
                'timestamp': metadata['timestamp'],
                'metrics': metadata['performance_metrics'],
                'file': metadata_file
            })
        
        return versions
    except Exception as e:
        return []

# ML Model Functions
def train_classification_models(X_train, y_train, X_test=None, y_test=None):
    """Train multiple classification models."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # Training performance
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Test performance if test data is available
        test_metrics = {}
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test)
            y_test_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            test_metrics = {
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'test_precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'test_recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'test_f1_score': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'test_roc_auc': roc_auc_score(y_test, y_test_pred_proba) if y_test_pred_proba is not None else None,
                'test_predictions': y_test_pred,
                'test_probabilities': y_test_pred_proba
            }
        
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_train, y_train_pred_proba) if y_train_pred_proba is not None else None,
            'predictions': y_train_pred,
            'probabilities': y_train_pred_proba,
            **test_metrics
        }
    
    return results

def perform_clustering(df_churned, feature_cols, n_clusters=4):
    """Perform K-means clustering on churned customers."""
    # Use available numeric features for clustering
    numeric_features = []
    for col in feature_cols:
        if col in df_churned.columns and df_churned[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
    
    if len(numeric_features) < 2:
        st.error("Not enough numeric features for clustering analysis.")
        return None, None, None, None
    
    X_cluster = df_churned[numeric_features].copy()
    
    # Handle any remaining missing values
    X_cluster = X_cluster.fillna(X_cluster.median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    return cluster_labels, kmeans, scaler, numeric_features

def find_association_rules_real(df_churned, categorical_cols, min_support=0.1):
    """Find association rules using available categorical features."""
    # Prepare data for association rules using available categorical columns
    transactions = []
    
    for _, row in df_churned.iterrows():
        transaction = []
        for col in categorical_cols:
            if col in df_churned.columns and pd.notna(row[col]):
                # Add categorical values as items
                transaction.append(f"{col}_{row[col]}")
        
        # Add binary features as items
        for col in df_churned.columns:
            if df_churned[col].dtype in ['int64', 'float64'] and df_churned[col].nunique() == 2:
                if row[col] == 1:
                    transaction.append(col)
        
        transactions.append(transaction)
    
    if not transactions or not any(transactions):
        return pd.DataFrame(), pd.DataFrame()
    
    # Convert to format needed for apriori
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Find frequent itemsets
    try:
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
            return rules, frequent_itemsets
        else:
            return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Error in association rules mining: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Genetic Algorithm
def genetic_algorithm_tuning(X, y, population_size=20, generations=10):
    """Use genetic algorithm to tune Random Forest hyperparameters."""
    
    # Define the problem
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except:
        pass  # Classes already exist
    
    toolbox = base.Toolbox()
    
    # Hyperparameter ranges
    toolbox.register("n_estimators", rand.randint, 50, 200)
    toolbox.register("max_depth", rand.randint, 3, 20)
    toolbox.register("min_samples_split", rand.randint, 2, 20)
    toolbox.register("min_samples_leaf", rand.randint, 1, 10)
    
    # Create individual
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.n_estimators, toolbox.max_depth, 
                     toolbox.min_samples_split, toolbox.min_samples_leaf), n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate_individual(individual):
        """Evaluate fitness of individual (hyperparameter set)."""
        n_est, max_d, min_split, min_leaf = individual
        
        try:
            rf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_split=min_split,
                min_samples_leaf=min_leaf,
                random_state=42
            )
            
            # Use cross-validation for fitness
            scores = cross_val_score(rf, X, y, cv=3, scoring='f1')
            return (np.mean(scores),)
        except:
            return (0.0,)
    
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run GA
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, 
                                  ngen=generations, stats=stats, 
                                  halloffame=hof, verbose=False)
    
    best_params = hof[0]
    return {
        'n_estimators': best_params[0],
        'max_depth': best_params[1],
        'min_samples_split': best_params[2],
        'min_samples_leaf': best_params[3]
    }, log

# Streamlit App
def main():
    st.markdown('<h1 class="main-header"> Intelligent CRM System for Customer Churn Prediction</h1>', 
                unsafe_allow_html=True)
    
    # Auto-load data on first run
    if not st.session_state.data_loaded:
        try:
            data_result = load_real_data()
            
            if data_result[0] is not None:
                combined_df, datasets = data_result
                
                # Preprocess the data
                df_processed = preprocess_real_data(combined_df)
                
                # Create engineered features
                df_final = create_engineered_features_real(df_processed)
                
                # Add simulated NLP features
                df_final = simulate_nlp_features_real(df_final)
                
                st.session_state.df = df_final
                st.session_state.datasets = datasets
                st.session_state.data_loaded = True
                
                # Auto-load models if available
                models, features, metadata = load_latest_models()
                if models:
                    st.session_state.models_loaded = True
                    st.session_state.model_trained = True
                    st.session_state.selected_features = features
                    st.session_state.model_info = metadata
                    
                    # Auto-generate predictions
                    try:
                        X = df_final[features].copy()
                        y = df_final['Churn'].copy()
                        
                        for col in X.columns:
                            X[col] = pd.to_numeric(X[col], errors='coerce')
                            X[col] = X[col].fillna(X[col].median())
                        
                        results = {}
                        for model_name, model in models.items():
                            y_pred = model.predict(X)
                            y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
                            
                            results[model_name] = {
                                'model': model,
                                'accuracy': accuracy_score(y, y_pred),
                                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                                'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0),
                                'roc_auc': roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else None,
                                'predictions': y_pred,
                                'probabilities': y_pred_proba
                            }
                        
                        st.session_state.model_results = results
                    except Exception as pred_error:
                        # Models will be loaded when user visits prediction page
                        st.session_state.model_results = None
                
                st.success(f"✓ System ready! Loaded {len(df_final):,} customer records")
            else:
                st.error("Failed to load data files from data/ directory")
                
        except Exception as e:
            st.error(f"Could not auto-load data: {str(e)}")
            st.info("Please check that data files exist in the data/ directory")
    
    # Sidebar
    st.sidebar.title(" Navigation")
    
    # Show data and model status in sidebar
    if st.session_state.data_loaded:
        st.sidebar.success(" Data Loaded")
        if 'df' in st.session_state.__dict__:
            st.sidebar.metric("Records", f"{len(st.session_state.df):,}")
    else:
        st.sidebar.warning(" No Data")
    
    if st.session_state.models_loaded:
        st.sidebar.success(" Models Ready")
    else:
        st.sidebar.info(" No Pre-trained Models")
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Choose a section:", [
        " Home",
        " Data Overview", 
        " Churn Prediction",
        " Customer Segmentation",
        " Association Rules",
        " AI Optimization",
        " Model Performance",
        " Interactive Prediction"
    ])
    
    # Quick action buttons in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
    
    if st.sidebar.button(" Reload Data", use_container_width=True):
        st.session_state.data_loaded = False
        st.rerun()
    
    if st.sidebar.button(" Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.session_state.model_trained = False
        st.session_state.clusters_created = False
        st.sidebar.success("Cache cleared!")
    
    if page == " Home":
        show_home_page()
    elif page == " Data Overview":
        show_data_overview()
    elif page == " Churn Prediction":
        show_churn_prediction()
    elif page == " Customer Segmentation":
        show_customer_segmentation()
    elif page == " Association Rules":
        show_association_rules()
    elif page == " AI Optimization":
        show_ai_optimization()
    elif page == " Model Performance":
        show_model_performance()
    elif page == " Interactive Prediction":
        show_interactive_prediction()

def show_home_page():
    # st.markdown("## Welcome to GRAHAK CRM System")
    
    # # System status overview
    # col1, col2, col3, col4 = st.columns(4)
    
    # with col1:
    #     if st.session_state.data_loaded and 'df' in st.session_state.__dict__:
    #         st.metric("Customer Records", f"{len(st.session_state.df):,}")
    #     else:
    #         st.metric("Customer Records", "0")
    
    # with col2:
    #     if st.session_state.models_loaded:
    #         st.metric("Model Status", "Ready", delta="Pre-trained")
    #     else:
    #         st.metric("Model Status", "Not Available", delta="Need training")
    
    # # Calculate best model info once for use in multiple columns
    # best_model_name = "N/A"
    # best_model_score = None
    
    # if st.session_state.models_loaded and 'model_info' in st.session_state.__dict__:
    #     # Use metadata from loaded models
    #     metadata = st.session_state.model_info
    #     if 'performance_metrics' in metadata:
    #         best_result = max(metadata['performance_metrics'].items(), 
    #                         key=lambda x: x[1].get('test_f1_score', x[1].get('f1_score', 0)))
    #         best_model_name = best_result[0].split()[0]
    #         best_model_score = best_result[1].get('test_f1_score', best_result[1].get('f1_score'))
    # elif st.session_state.model_trained and 'model_results' in st.session_state.__dict__:
    #     # Use current session model results
    #     best_result = max(st.session_state.model_results.items(), 
    #                      key=lambda x: x[1].get('test_f1_score', x[1].get('f1_score', 0)))
    #     best_model_name = best_result[0].split()[0]
    #     best_model_score = best_result[1].get('test_f1_score', best_result[1].get('f1_score'))
    
    # with col3:
    #     st.metric("Best Model", best_model_name)
    
    # with col4:
    #     if best_model_score is not None:
    #         st.metric("F1-Score", f"{best_model_score:.3f}")
    #     else:
    #         st.metric("F1-Score", "N/A")
    
    # Data summary
    if st.session_state.data_loaded and 'df' in st.session_state.__dict__:
        df = st.session_state.df
        
        st.subheader(" Customer Data Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        
        with col2:
            if 'Churn' in df.columns:
                churn_rate = df['Churn'].mean()
                st.metric("Churn Rate", f"{churn_rate:.1%}")
        
        with col3:
            if 'Monthly Charge' in df.columns:
                avg_revenue = df['Monthly Charge'].mean()
                st.metric("Avg Revenue", f"${avg_revenue:.0f}")
        
        with col4:
            if 'Tenure in Months' in df.columns:
                avg_tenure = df['Tenure in Months'].mean()
                st.metric("Avg Tenure", f"{avg_tenure:.0f}mo")
        
        with col5:
            if 'Satisfaction Score' in df.columns:
                avg_satisfaction = df['Satisfaction Score'].mean()
                st.metric("Satisfaction", f"{avg_satisfaction:.1f}/5")
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        
        Your system is **ready to use** with:
        
        **Instant Features:**
        - **Pre-loaded Data**: Customer data loaded automatically
        - **Pre-trained Models**: High-accuracy models ready for predictions
        - **Real-time Analysis**: Instant churn probability assessment
        - **Customer Segmentation**: Automatic clustering of at-risk customers
        - **Pattern Discovery**: Association rules for actionable insights
        - **Comprehensive Reports**: Detailed analytics and visualizations
        
        **Quick Actions:**
        - **View Predictions**: Go to "Churn Prediction" to see instant results
        - **Analyze Segments**: Check "Customer Segmentation" for groups
        - **Find Patterns**: Use "Association Rules" for insights
        - **Test Individual**: Try "Interactive Prediction" for single customers
        - **Export Results**: Download predictions for CRM integration
        """)
        
        if st.session_state.model_trained:
            st.success(" System ready! Navigate to any section to start analyzing.")
        else:
            st.info(" Data loaded. Visit 'Churn Prediction' to generate predictions.")
    
    with col2:
        st.markdown("""
        <div class="success-card">
        <h3> System Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        status_items = []
        if st.session_state.data_loaded:
            status_items.append("✓ Data loaded automatically")
        else:
            status_items.append("✗ Data not loaded")
        
        if st.session_state.models_loaded:
            status_items.append("✓ Pre-trained models available")
        else:
            status_items.append("✗ No pre-trained models")
        
        if st.session_state.model_trained:
            status_items.append("✓ Predictions ready")
        else:
            status_items.append("⚠ Predictions not generated")
        
        for item in status_items:
            st.write(item)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if not st.session_state.models_loaded:
            st.warning("Train models in 'Churn Prediction' section")
    
    # Performance metrics if available
    if st.session_state.model_trained and 'model_results' in st.session_state.__dict__:
        st.markdown("---")
        st.subheader(" Model Performance on Current Data")
        
        results = st.session_state.model_results
        comparison_data = []
        
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.3f}",
                'Precision': f"{result['precision']:.3f}",
                'Recall': f"{result['recall']:.3f}",
                'F1-Score': f"{result['f1_score']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Quick actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(" View Detailed Predictions", use_container_width=True, type="primary"):
                st.session_state.page = " Churn Prediction"
                st.rerun()
        
        with col2:
            if st.button(" Analyze Customer Segments", use_container_width=True):
                st.session_state.page = " Customer Segmentation"
                st.rerun()
        
        with col3:
            if st.button(" Export Predictions", use_container_width=True):
                predictions_df = st.session_state.df.copy()
                for name, result in results.items():
                    predictions_df[f'{name}_Prediction'] = result['predictions']
                    if result['probabilities'] is not None:
                        predictions_df[f'{name}_Probability'] = result['probabilities']
                
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label=" Download Predictions CSV",
                    data=csv,
                    file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

def show_data_overview():
    st.markdown('<h2 class="sub-header"> Real Data Overview & Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning(" Please load real data first from the Home page!")
        return
    
    df = st.session_state.df
    datasets = st.session_state.datasets
    
    # Dataset Info
    st.subheader(" Dataset Information")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Column Names:**")
        st.write(df.columns.tolist())
        
        # Data types
        st.write("**Data Types:**")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(dtype_df)
    
    with col2:
        st.subheader(" Key Metrics")
        st.metric("Total Samples", len(df))
        st.metric("Total Features", len(df.columns))
        st.metric("Missing Values", df.isnull().sum().sum())
        st.metric("Duplicate Rows", df.duplicated().sum())
        
        # Find and display target distribution
        target_col = 'Churn'
        
        if target_col in df.columns:
            try:
                churn_values = df[target_col].copy()
                
                # Convert to numeric if not already
                churn_values = pd.to_numeric(churn_values, errors='coerce').fillna(0)
                churn_rate = churn_values.mean()
                st.metric("Churn Rate", f"{churn_rate:.2%}")
            except Exception as e:
                st.error(f"Error calculating churn rate: {str(e)}")
                st.metric("Churn Rate", "Error")
        
        # Show test data info if available
        if st.session_state.test_data_loaded and 'test' in datasets:
            test_df = datasets['test']
            st.metric("Test Samples", len(test_df))
            if target_col in test_df.columns:
                test_churn_rate = test_df[target_col].mean()
                st.metric("Test Churn Rate", f"{test_churn_rate:.2%}")
    
    # Statistical Summary
    st.subheader(" Statistical Summary")
    try:
        # Get numeric columns more carefully
        numeric_cols = []
        
        for col in df.columns:
            try:
                # Try to convert to numeric and check if it's actually numeric
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                # Skip columns that can't be converted to numeric
                continue
        
        if len(numeric_cols) > 0:
            # Create a clean numeric dataframe
            numeric_df = df[numeric_cols].copy()
            
            # Ensure all columns are properly numeric
            for col in numeric_cols:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
            
            # Remove any rows with NaN values that might have been created
            numeric_df = numeric_df.dropna()
            
            if len(numeric_df) > 0:
                st.dataframe(numeric_df.describe())
            else:
                st.warning("No valid numeric data found after cleaning.")
        else:
            st.info("No numeric columns found for statistical summary.")
    except Exception as e:
        st.error(f"Error generating statistical summary: {str(e)}")
        st.info("This might be due to data type conversion issues. Please check your data integrity.")
    
    # Data Quality Check
    st.subheader(" Data Quality Assessment")
    quality_issues = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for potential data corruption in string columns
            max_length = df[col].astype(str).str.len().max()
            if max_length > 100:
                quality_issues.append(f" Column '{col}' has unusually long values (max: {max_length} chars)")
            
            # Check for mixed data types
            unique_types = set(type(x).__name__ for x in df[col].dropna())
            if len(unique_types) > 1:
                quality_issues.append(f" Column '{col}' has mixed data types: {unique_types}")
    
    if quality_issues:
        st.warning("Data quality issues detected:")
        for issue in quality_issues:
            st.write(issue)
    else:
        st.success(" No major data quality issues detected!")
    
    # Feature Distribution Plots
    st.subheader(" Feature Distributions")
    
    # Select columns for visualization
    try:
        # Get numeric columns more carefully
        safe_numeric_cols = []
        
        for col in df.columns:
            try:
                # Try to convert to numeric and check if it's actually numeric
                test_series = pd.to_numeric(df[col], errors='raise')
                # If successful, it's a safe numeric column
                safe_numeric_cols.append(col)
            except (ValueError, TypeError):
                # Skip columns that can't be converted to numeric
                continue
        
        if safe_numeric_cols:
            viz_cols = st.multiselect("Select columns to visualize:", 
                                     safe_numeric_cols,
                                     default=safe_numeric_cols[:4] if len(safe_numeric_cols) >= 4 else safe_numeric_cols)
            
            if viz_cols:
                try:
                    cols_per_row = 2
                    for i in range(0, len(viz_cols), cols_per_row):
                        row_cols = viz_cols[i:i+cols_per_row]
                        chart_cols = st.columns(len(row_cols))
                        
                        for j, col in enumerate(row_cols):
                            with chart_cols[j]:
                                try:
                                    # Ensure data is clean for visualization
                                    clean_data = df[[col]].dropna()
                                    
                                    if len(clean_data) == 0:
                                        st.warning(f"No valid data for column '{col}'")
                                        continue
                                    
                                    if clean_data[col].nunique() <= 10:  # Categorical-like numeric
                                        fig = px.histogram(clean_data, x=col, title=f'{col} Distribution')
                                    else:  # Continuous numeric
                                        fig = px.box(clean_data, y=col, title=f'{col} Distribution')
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating chart for '{col}': {str(e)}")
                except Exception as e:
                    st.error(f"Error in visualization section: {str(e)}")
        else:
            st.warning("No suitable numeric columns found for visualization.")
    except Exception as e:
        st.error(f"Error in feature distributions: {str(e)}")
    
    # Correlation Analysis (for numeric features only)
    try:
        if len(safe_numeric_cols) > 1:
            st.subheader(" Feature Correlations")
            
            # Create clean numeric dataframe for correlation
            correlation_df = df[safe_numeric_cols].copy()
            
            # Ensure all columns are properly numeric
            for col in safe_numeric_cols:
                correlation_df[col] = pd.to_numeric(correlation_df[col], errors='coerce')
            
            # Remove any rows with NaN values
            correlation_df = correlation_df.dropna()
            
            if len(correlation_df) > 0 and len(correlation_df.columns) > 1:
                corr_matrix = correlation_df.corr()
                
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                title="Feature Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient numeric data for correlation analysis.")
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
    
    # Data Sample
    st.subheader(" Sample Data")
    try:
        # Show a clean sample of the data
        sample_df = df.head(10)
        
        # Truncate any overly long string values for display
        display_df = sample_df.copy()
        for col in display_df.select_dtypes(include=['object']).columns:
            display_df[col] = display_df[col].astype(str).apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
        
        st.dataframe(display_df)
    except Exception as e:
        st.error(f"Error displaying data sample: {str(e)}")

def show_churn_prediction():
    st.markdown('<h2 class="sub-header"> Churn Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning(" Please load customer data first from the Home page!")
        return
    
    df = st.session_state.df
    target_col = 'Churn'
    
    if target_col not in df.columns:
        st.error(f" Target column '{target_col}' not found in the dataset!")
        return
    
    # Check for pre-trained models
    models, features, metadata = load_latest_models()
    use_pretrained = False
    
    if models and features:
        st.success(" Using pre-trained models for instant predictions")
        
        # Display model info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Models Available", len(models))
        with col2:
            st.metric("Last Trained", metadata['timestamp'][:8])
        with col3:
            best_model = max(metadata['performance_metrics'].items(), 
                           key=lambda x: x[1].get('test_f1_score', x[1]['f1_score']))
            st.metric("Best Model", best_model[0].split()[0])
        with col4:
            best_score = best_model[1].get('test_f1_score', best_model[1]['f1_score'])
            st.metric("Best F1-Score", f"{best_score:.3f}")
        
        # Use pre-trained models
        use_pretrained = st.checkbox(" Use pre-trained models (Recommended for speed)", value=True)
        
        if use_pretrained:
            st.info(" Getting predictions from pre-trained models...")
            
            try:
                # Prepare features
                X = df[features].copy()
                y = df[target_col].copy()
                
                # Clean data
                for col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(X[col].median())
                
                # Get predictions from all models
                results = {}
                for model_name, model in models.items():
                    y_pred = model.predict(X)
                    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    results[model_name] = {
                        'model': model,
                        'accuracy': accuracy_score(y, y_pred),
                        'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                        'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0),
                        'roc_auc': roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else None,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba
                    }
                
                st.session_state.model_results = results
                st.session_state.model_trained = True
                st.session_state.selected_features = features
                
                st.success(" Predictions generated successfully!")
                
                # Show quick metrics
                st.subheader(" Model Performance on Current Data")
                cols = st.columns(len(results))
                for idx, (name, result) in enumerate(results.items()):
                    with cols[idx]:
                        st.metric(name, f"{result['accuracy']:.3f}", 
                                delta=f"F1: {result['f1_score']:.3f}")
                
            except Exception as e:
                st.error(f" Error using pre-trained models: {str(e)}")
                st.info("Features in current data might not match trained models. Please retrain.")
                use_pretrained = False
    else:
        use_pretrained = False
        st.info(" No pre-trained models found. Training new models...")
    
    # Option to retrain models
    if not use_pretrained or st.checkbox(" Retrain models for better accuracy", value=False):
        st.subheader(" Model Training Configuration")
        
        # Feature selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        non_predictive_cols = ['Customer ID', 'Zip Code', 'Latitude', 'Longitude', 'Population']
        numeric_cols = [col for col in numeric_cols if col not in non_predictive_cols]
        
        categorical_encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
        all_potential_features = numeric_cols + categorical_encoded_cols
        
        recommended_features = [
            'Tenure in Months', 'Monthly Charge', 'Total Charges', 'Total Revenue',
            'Age', 'Number of Dependents', 'Number of Referrals', 'Satisfaction Score',
            'CLTV', 'Churn Score', 'TotalServices', 'HighSpender'
        ]
        
        existing_recommended_features = [col for col in recommended_features if col in all_potential_features]
        
        # Use existing features if available, otherwise use recommended
        default_features = features if features else existing_recommended_features
        
        selected_features = st.multiselect(
            "Select features for modeling:", 
            all_potential_features,
            default=default_features
        )
        
        if not selected_features:
            st.warning("Please select at least one feature for modeling.")
            return
        
        st.session_state.selected_features = selected_features
        
        # Test data configuration
        datasets = st.session_state.datasets if 'datasets' in st.session_state.__dict__ else {}
        if st.session_state.test_data_loaded and 'test' in datasets:
            use_test_data = st.checkbox("Use separate test dataset", value=True)
            if use_test_data:
                test_df = datasets['test']
                test_df_processed = preprocess_real_data(test_df)
                test_df_final = create_engineered_features_real(test_df_processed)
                X_test = test_df_final[selected_features].copy()
                y_test = test_df_final[target_col].copy()
                for col in X_test.columns:
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                    X_test[col] = X_test[col].fillna(X_test[col].median())
        else:
            use_test_data = False
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        
        if st.button(" Train New Models", type="primary"):
            with st.spinner("Training models with optimized parameters..."):
                # Prepare data
                X = df[selected_features].copy()
                y = df[target_col].copy()
                
                for col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(X[col].median())
                
                # Split or use test data
                if use_test_data:
                    X_train, y_train = X, y
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                
                # Train models
                results = train_classification_models(X_train, y_train, X_test, y_test)
                
                # Save models
                model_info = {
                    'train_samples': len(X_train),
                    'test_samples': len(X_test) if use_test_data or not use_test_data else 0,
                    'features_count': len(selected_features),
                    'target_column': target_col
                }
                
                success, timestamp = save_models(results, selected_features, model_info)
                
                if success:
                    st.success(f" Models trained and saved successfully! (Version: {timestamp})")
                    st.session_state.model_results = results
                    st.session_state.model_trained = True
                    st.session_state.models_loaded = True
                else:
                    st.warning(" Models trained but not saved. Using in-memory models.")
                    st.session_state.model_results = results
                    st.session_state.model_trained = True
    
    # Display results if models are trained
    if st.session_state.model_trained:
        results = st.session_state.model_results
        
        # Model Comparison
        st.subheader(" Model Performance Comparison")
        
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Train Accuracy': result['accuracy'],
                'Test Accuracy': result.get('test_accuracy', 'N/A'),
                'Train F1-Score': result['f1_score'],
                'Test F1-Score': result.get('test_f1_score', 'N/A'),
                'Train Precision': result['precision'],
                'Test Precision': result.get('test_precision', 'N/A'),
                'Train Recall': result['recall'],
                'Test Recall': result.get('test_recall', 'N/A'),
                'Train ROC-AUC': result['roc_auc'] if result['roc_auc'] else 'N/A',
                'Test ROC-AUC': result.get('test_roc_auc', 'N/A') if result.get('test_roc_auc') else 'N/A'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Find best test accuracy (excluding N/A values)
            test_acc_values = comparison_df[comparison_df['Test Accuracy'] != 'N/A']['Test Accuracy'].astype(float)
            if len(test_acc_values) > 0:
                best_acc_idx = test_acc_values.idxmax()
                best_acc = comparison_df.loc[best_acc_idx]
                st.metric(" Best Test Accuracy", f"{best_acc['Test Accuracy']:.3f}", 
                         delta=f"Model: {best_acc['Model']}")
        
        with col2:
            # Find best test F1-Score (excluding N/A values)
            test_f1_values = comparison_df[comparison_df['Test F1-Score'] != 'N/A']['Test F1-Score'].astype(float)
            if len(test_f1_values) > 0:
                best_f1_idx = test_f1_values.idxmax()
                best_f1 = comparison_df.loc[best_f1_idx]
                st.metric(" Best Test F1-Score", f"{best_f1['Test F1-Score']:.3f}", 
                         delta=f"Model: {best_f1['Model']}")
        
        with col3:
            # Find best test ROC-AUC (excluding N/A values)
            test_roc_values = comparison_df[comparison_df['Test ROC-AUC'] != 'N/A']['Test ROC-AUC'].astype(float)
            if len(test_roc_values) > 0:
                best_roc_idx = test_roc_values.idxmax()
                best_roc = comparison_df.loc[best_roc_idx]
                st.metric(" Best Test ROC-AUC", f"{best_roc['Test ROC-AUC']:.3f}", 
                         delta=f"Model: {best_roc['Model']}")
        
        # Detailed comparison table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Model Performance Visualization
        fig = px.bar(comparison_df.melt(id_vars=['Model'], 
                                      value_vars=['Train Accuracy', 'Test Accuracy', 
                                                 'Train F1-Score', 'Test F1-Score']),
                    x='Model', y='value', color='variable',
                    title='Model Performance Comparison (Train vs Test)',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance (Random Forest)
        if 'Random Forest' in results:
            st.subheader(" Feature Importance (Random Forest)")
            rf_model = results['Random Forest']['model']
            
            # Use session state selected_features which should always be set
            feature_list = st.session_state.selected_features if st.session_state.selected_features else []
            
            if feature_list and hasattr(rf_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': feature_list,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance.head(15), x='Importance', y='Feature', 
                            orientation='h', title='Top 15 Feature Importances')
                st.plotly_chart(fig, use_container_width=True)

# ... (rest of the functions remain mostly the same, but need to update show_model_performance to handle test data)

def show_model_performance():
    st.markdown('<h2 class="sub-header"> Comprehensive Model Performance Analysis</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning(" Please train models first in the Churn Prediction section!")
        return
    
    results = st.session_state.model_results
    df = st.session_state.df
    target_col = st.session_state.target_column
    
    # Model Selection
    st.subheader(" Select Model for Detailed Analysis")
    selected_model = st.selectbox("Choose Model:", list(results.keys()))
    
    model_result = results[selected_model]
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(" Train Accuracy", f"{model_result['accuracy']:.3f}")
        if 'test_accuracy' in model_result and model_result['test_accuracy'] != 'N/A':
            st.metric(" Test Accuracy", f"{model_result['test_accuracy']:.3f}")
    
    with col2:
        st.metric(" Train F1-Score", f"{model_result['f1_score']:.3f}")
        if 'test_f1_score' in model_result and model_result['test_f1_score'] != 'N/A':
            st.metric(" Test F1-Score", f"{model_result['test_f1_score']:.3f}")
    
    with col3:
        st.metric(" Train Precision", f"{model_result['precision']:.3f}")
        if 'test_precision' in model_result and model_result['test_precision'] != 'N/A':
            st.metric(" Test Precision", f"{model_result['test_precision']:.3f}")
    
    with col4:
        st.metric(" Train Recall", f"{model_result['recall']:.3f}")
        if 'test_recall' in model_result and model_result['test_recall'] != 'N/A':
            st.metric(" Test Recall", f"{model_result['test_recall']:.3f}")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Training Confusion Matrix")
        cm_train = confusion_matrix(df[target_col], model_result['predictions'])
        
        fig = px.imshow(cm_train, text_auto=True, aspect="auto",
                       labels=dict(x="Predicted", y="Actual"),
                       x=['No Churn', 'Churn'],
                       y=['No Churn', 'Churn'],
                       title=f'Training Confusion Matrix - {selected_model}')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'test_predictions' in model_result and model_result['test_predictions'] is not None:
            st.subheader(" Test Confusion Matrix")
            cm_test = confusion_matrix(st.session_state.y_test, model_result['test_predictions'])
            
            fig = px.imshow(cm_test, text_auto=True, aspect="auto",
                           labels=dict(x="Predicted", y="Actual"),
                           x=['No Churn', 'Churn'],
                           y=['No Churn', 'Churn'],
                           title=f'Test Confusion Matrix - {selected_model}')
            st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Training ROC Curve")
        if model_result['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(df[target_col], model_result['probabilities'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {model_result["roc_auc"]:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier'))
            fig.update_layout(title='Training ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'test_probabilities' in model_result and model_result['test_probabilities'] is not None:
            st.subheader(" Test ROC Curve")
            fpr_test, tpr_test, _ = roc_curve(st.session_state.y_test, model_result['test_probabilities'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr_test, y=tpr_test, name=f'ROC Curve (AUC = {model_result["test_roc_auc"]:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier'))
            fig.update_layout(title='Test ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Classification Report
    st.subheader(" Detailed Training Classification Report")
    
    report_train = classification_report(df[target_col], model_result['predictions'], 
                                       target_names=['No Churn', 'Churn'], output_dict=True)
    
    report_train_df = pd.DataFrame(report_train).transpose()
    st.dataframe(report_train_df.round(3), use_container_width=True)
    
    if 'test_predictions' in model_result and model_result['test_predictions'] is not None:
        st.subheader(" Detailed Test Classification Report")
        
        report_test = classification_report(st.session_state.y_test, model_result['test_predictions'], 
                                          target_names=['No Churn', 'Churn'], output_dict=True)
        
        report_test_df = pd.DataFrame(report_test).transpose()
        st.dataframe(report_test_df.round(3), use_container_width=True)
    
    # Model Comparison Summary
    st.subheader(" Model Ranking (Based on Test Performance)")
    
    ranking_data = []
    for name, result in results.items():
        if 'test_accuracy' in result and result['test_accuracy'] != 'N/A':
            score = (result['test_accuracy'] + result['test_f1_score'] + 
                    result['test_precision'] + result['test_recall']) / 4
            ranking_data.append({
                'Model': name,
                'Overall Score': score,
                'Test Accuracy': result['test_accuracy'],
                'Test F1-Score': result['test_f1_score'],
                'Test Precision': result['test_precision'],
                'Test Recall': result['test_recall']
            })
    
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data).sort_values('Overall Score', ascending=False)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        st.dataframe(ranking_df, use_container_width=True)
    else:
        st.info("No test performance metrics available for ranking.")

def show_customer_segmentation():
    st.markdown('<h2 class="sub-header"> Customer Segmentation Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning(" Please load real data first from the Home page!")
        return
    
    df = st.session_state.df
    
    # Find target column
    target_col = 'Churn'
    
    if target_col not in df.columns:
        st.error(f" Target column '{target_col}' not found for segmentation!")
        return
    
    df_churned = df[df[target_col] == 1].copy()
    
    if len(df_churned) == 0:
        st.error("No churned customers found in the dataset!")
        return
    
    st.info(f" Analyzing {len(df_churned)} churned customers for segmentation")
    
    # Feature selection for clustering
    numeric_cols = df_churned.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Remove ID columns and other non-predictive columns
    non_predictive_cols = ['Customer ID', 'Zip Code', 'Latitude', 'Longitude', 'Population']
    numeric_cols = [col for col in numeric_cols if col not in non_predictive_cols]
    
    # Recommended features for clustering
    recommended_cluster_features = [
        'Tenure in Months', 'Monthly Charge', 'Total Charges', 'Total Revenue',
        'Age', 'Number of Dependents', 'Number of Referrals', 'Satisfaction Score',
        'CLTV', 'Churn Score', 'TotalServices'
    ]
    
    # Filter to only include features that actually exist in the dataset
    existing_cluster_features = [col for col in recommended_cluster_features if col in numeric_cols]
    
    cluster_features = st.multiselect("Select features for clustering:", 
                                    numeric_cols,
                                    default=existing_cluster_features[:6] if len(existing_cluster_features) > 6 else existing_cluster_features)
    
    if not cluster_features:
        st.warning("Please select at least 2 features for clustering.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_clusters = st.slider(" Number of Clusters", min_value=2, max_value=8, value=4)
    
    with col2:
        if st.button(" Perform Customer Segmentation", type="primary"):
            with st.spinner("Performing customer segmentation..."):
                cluster_result = perform_clustering(df_churned, cluster_features, n_clusters)
                
                if cluster_result[0] is not None:
                    cluster_labels, kmeans_model, scaler, used_features = cluster_result
                    
                    # Add cluster labels to dataframe
                    df_churned_clustered = df_churned.copy()
                    df_churned_clustered['ClusterLabel'] = cluster_labels
                    
                    st.session_state.df_churned_clustered = df_churned_clustered
                    st.session_state.cluster_features = used_features
                    st.session_state.clusters_created = True
                    
                    st.success(" Customer segmentation completed!")
    
    if st.session_state.clusters_created:
        df_clustered = st.session_state.df_churned_clustered
        used_features = st.session_state.cluster_features
        
        # Cluster Overview
        st.subheader(" Cluster Overview")
        
        # Create summary statistics for available features
        summary_cols = [col for col in used_features if col in df_clustered.columns]
        if summary_cols:
            cluster_summary = df_clustered.groupby('ClusterLabel')[summary_cols].mean().round(2)
            st.dataframe(cluster_summary)
        
        # Cluster Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster size distribution
            cluster_counts = df_clustered['ClusterLabel'].value_counts().sort_index()
            fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                        title='Customer Count by Cluster',
                        labels={'x': 'Cluster', 'y': 'Number of Customers'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cluster visualization (2D projection)
            if len(used_features) >= 2:
                fig = px.scatter(df_clustered, x=used_features[0], y=used_features[1], 
                               color='ClusterLabel', title='Customer Clusters (2D Projection)')
                st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Characteristics
        st.subheader(" Cluster Characteristics")
        
        for cluster_id in sorted(df_clustered['ClusterLabel'].unique()):
            cluster_data = df_clustered[df_clustered['ClusterLabel'] == cluster_id]
            
            with st.expander(f" Cluster {cluster_id} ({len(cluster_data)} customers)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Key Statistics:**")
                    for feature in used_features[:5]:  # Show top 5 features
                        if feature in cluster_data.columns:
                            avg_val = cluster_data[feature].mean()
                            st.write(f"- Avg {feature}: {avg_val:.2f}")
                
                with col2:
                    # Show a sample of customers in this cluster
                    st.write("**Sample Customers:**")
                    sample_cols = ['Customer ID'] + used_features[:3]
                    available_cols = [col for col in sample_cols if col in cluster_data.columns]
                    if available_cols:
                        st.dataframe(cluster_data[available_cols].head(3))

def show_association_rules():
    st.markdown('<h2 class="sub-header"> Association Rules Mining</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning(" Please load real data first from the Home page!")
        return
    
    df = st.session_state.df
    
    # Find target column
    target_col = 'Churn'
    
    if target_col not in df.columns:
        st.error(f" Target column '{target_col}' not found!")
        return
    
    df_churned = df[df[target_col] == 1].copy()
    
    if len(df_churned) == 0:
        st.error("No churned customers found in the dataset!")
        return
    
    # Get categorical columns for association rules
    categorical_cols = df_churned.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if 'id' not in col.lower() and col != 'Lat Long']
    
    # Recommended categorical features
    recommended_categorical = [
        'Contract', 'Payment Method', 'Internet Type', 'Offer', 
        'Churn Category', 'Churn Reason', 'City'
    ]
    
    # Filter to only include features that actually exist in the dataset
    existing_categorical = [col for col in recommended_categorical if col in categorical_cols]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        min_support = st.slider(" Minimum Support", 0.05, 0.5, 0.1, 0.05)
        min_confidence = st.slider(" Minimum Confidence", 0.3, 0.9, 0.5, 0.1)
        
        st.write(f"**Available categorical features:** {len(categorical_cols)}")
        if categorical_cols:
            st.write("Sample features:", categorical_cols[:5])
    
    with col2:
        if st.button(" Discover Association Rules", type="primary"):
            with st.spinner("Mining association rules..."):
                rules, frequent_itemsets = find_association_rules_real(df_churned, existing_categorical, min_support)
                
                if not rules.empty:
                    st.session_state.association_rules = rules
                    st.session_state.frequent_itemsets = frequent_itemsets
                    st.success(f" Found {len(rules)} association rules!")
                else:
                    st.warning("No association rules found with the current parameters. Try lowering the support threshold.")
    
    if 'association_rules' in st.session_state and len(st.session_state.association_rules) > 0:
        rules_df = st.session_state.association_rules
        
        st.subheader(" Discovered Association Rules")
        
        # Display rules in a more readable format
        display_rules = []
        for idx, rule in rules_df.iterrows():
            display_rules.append({
                'Rule': f"{list(rule['antecedents'])} → {list(rule['consequents'])}",
                'Support': f"{rule['support']:.3f}",
                'Confidence': f"{rule['confidence']:.3f}",
                'Lift': f"{rule['lift']:.3f}"
            })
        
        display_df = pd.DataFrame(display_rules)
        st.dataframe(display_df, use_container_width=True)
        
        # Rule visualization
        if len(rules_df) > 0:
            fig = px.scatter(rules_df, x='support', y='confidence', 
                           size='lift', hover_data=['lift'],
                           title='Association Rules: Support vs Confidence (Size = Lift)')
            st.plotly_chart(fig, use_container_width=True)

def show_ai_optimization():
    st.markdown('<h2 class="sub-header"> AI-Powered Hyperparameter Optimization</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning(" Please load real data first from the Home page!")
        return
    
    if not st.session_state.model_trained:
        st.warning(" Please train models first in the Churn Prediction section!")
        return
    
    df = st.session_state.df
    
    st.subheader(" Genetic Algorithm Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        population_size = st.slider(" Population Size", 10, 50, 20)
    with col2:
        generations = st.slider(" Generations", 5, 20, 10)
    with col3:
        st.write("**Target Model:** Random Forest")
        st.write("**Optimization Metric:** F1-Score")
    
    if st.button(" Start Genetic Algorithm Optimization", type="primary"):
        with st.spinner("Running genetic algorithm optimization..."):
            # Get the data used for training
            X = df[st.session_state.selected_features].fillna(df[st.session_state.selected_features].median())
            y = df[st.session_state.target_column]
            
            # Run genetic algorithm
            best_params, log = genetic_algorithm_tuning(X, y, population_size, generations)
            
            st.session_state.ga_best_params = best_params
            st.session_state.ga_log = log
            
            st.success(" Genetic Algorithm optimization completed!")
            
            # Train model with optimized parameters
            optimized_rf = RandomForestClassifier(**best_params, random_state=42)
            optimized_rf.fit(X, y)
            y_pred_opt = optimized_rf.predict(X)
            
            # Compare with original Random Forest
            original_rf = st.session_state.model_results['Random Forest']
            
            st.subheader(" Optimization Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Random Forest:**")
                st.write(f"- Accuracy: {original_rf['accuracy']:.3f}")
                st.write(f"- F1-Score: {original_rf['f1_score']:.3f}")
                st.write(f"- Precision: {original_rf['precision']:.3f}")
                st.write(f"- Recall: {original_rf['recall']:.3f}")
            
            with col2:
                st.write("**Optimized Random Forest:**")
                opt_accuracy = accuracy_score(y, y_pred_opt)
                opt_f1 = f1_score(y, y_pred_opt, average='weighted', zero_division=0)
                opt_precision = precision_score(y, y_pred_opt, average='weighted', zero_division=0)
                opt_recall = recall_score(y, y_pred_opt, average='weighted', zero_division=0)
                
                st.write(f"- Accuracy: {opt_accuracy:.3f}")
                st.write(f"- F1-Score: {opt_f1:.3f}")
                st.write(f"- Precision: {opt_precision:.3f}")
                st.write(f"- Recall: {opt_recall:.3f}")
            
            st.write("**Optimized Parameters:**")
            st.json(best_params)

def show_interactive_prediction():
    st.markdown('<h2 class="sub-header"> Interactive Churn Prediction</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning(" Please train models first in the Churn Prediction section!")
        return
    
    df = st.session_state.df
    feature_names = st.session_state.selected_features
    target_col = st.session_state.target_column
    
    st.subheader(" Enter Customer Information")
    st.info("Adjust the sliders below based on your real dataset features")
    
    # Create dynamic input form based on selected features
    input_values = {}
    
    # Organize features into columns
    num_cols = 3
    features_per_col = len(feature_names) // num_cols + (1 if len(feature_names) % num_cols != 0 else 0)
    
    cols = st.columns(num_cols)
    
    for i, feature in enumerate(feature_names):
        col_idx = i // features_per_col
        if col_idx >= num_cols:
            col_idx = num_cols - 1
        
        with cols[col_idx]:
            if feature in df.columns:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                
                input_values[feature] = st.slider(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"slider_{feature}"
                )
    
    if st.button(" Predict Churn Probability", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame([input_values])
        
        # Make predictions with all models
        st.subheader(" Prediction Results")
        
        results = st.session_state.model_results
        
        for model_name, model_result in results.items():
            model = model_result['model']
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**{model_name}**")
            
            with col2:
                if prediction == 1:
                    st.error(" Likely to Churn")
                else:
                    st.success(" Likely to Stay")
            
            with col3:
                if probability is not None:
                    churn_prob = probability[1]
                    st.write(f"Churn Probability: {churn_prob:.2%}")
                    
                    # Color-coded probability
                    if churn_prob > 0.7:
                        st.error(f"High Risk: {churn_prob:.2%}")
                    elif churn_prob > 0.4:
                        st.warning(f"Medium Risk: {churn_prob:.2%}")
                    else:
                        st.success(f"Low Risk: {churn_prob:.2%}")
        
        # Show input summary
        st.subheader(" Customer Profile Summary")
        
        profile_df = pd.DataFrame(list(input_values.items()), columns=['Feature', 'Value'])
        st.dataframe(profile_df, use_container_width=True)
def show_interactive_prediction():
    st.markdown('<h2 class="sub-header"> Interactive Churn Prediction</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning(" Please train models first in the Churn Prediction section!")
        return
    
    df = st.session_state.df
    feature_names = st.session_state.selected_features
    target_col = st.session_state.target_column
    
    st.subheader(" Enter Customer Information")
    st.info("Adjust the sliders below based on your real dataset features")
    
    # Create dynamic input form based on selected features
    input_values = {}
    
    # Organize features into columns
    num_cols = 3
    features_per_col = len(feature_names) // num_cols + (1 if len(feature_names) % num_cols != 0 else 0)
    
    cols = st.columns(num_cols)
    
    for i, feature in enumerate(feature_names):
        col_idx = i // features_per_col
        if col_idx >= num_cols:
            col_idx = num_cols - 1
        
        with cols[col_idx]:
            if feature in df.columns:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                
                input_values[feature] = st.slider(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"slider_{feature}"
                )
    
    if st.button(" Predict Churn Probability", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame([input_values])
        
        # Make predictions with all models
        st.subheader(" Prediction Results")
        
        results = st.session_state.model_results
        
        for model_name, model_result in results.items():
            model = model_result['model']
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**{model_name}**")
            
            with col2:
                if prediction == 1:
                    st.error(" Likely to Churn")
                else:
                    st.success(" Likely to Stay")
            
            with col3:
                if probability is not None:
                    churn_prob = probability[1]
                    st.write(f"Churn Probability: {churn_prob:.2%}")
                    
                    # Color-coded probability
                    if churn_prob > 0.7:
                        st.error(f"High Risk: {churn_prob:.2%}")
                    elif churn_prob > 0.4:
                        st.warning(f"Medium Risk: {churn_prob:.2%}")
                    else:
                        st.success(f"Low Risk: {churn_prob:.2%}")
        
        # Show input summary
        st.subheader(" Customer Profile Summary")
        
        profile_df = pd.DataFrame(list(input_values.items()), columns=['Feature', 'Value'])
        st.dataframe(profile_df, use_container_width=True)

if __name__ == "__main__":
    main()