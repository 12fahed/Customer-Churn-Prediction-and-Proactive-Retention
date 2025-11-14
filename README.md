#  GRAHAK - CRM Simulator for Churn Prediction & Analysis

A comprehensive machine learning application that predicts customer churn and provides proactive retention strategies using advanced AI techniques.

##  Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Architecture](#technical-architecture)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

##  Overview

**GRAHAK** is an intelligent Customer Relationship Management (CRM) system designed to predict customer churn and enable proactive retention strategies. Built using Streamlit and advanced machine learning algorithms, this system analyzes customer behavior patterns to identify at-risk customers and provides actionable insights for business decision-making.

### Key Objectives
- **Predict Customer Churn**: Identify customers likely to leave before they actually do
- **Customer Segmentation**: Group customers based on behavior and characteristics
- **Pattern Discovery**: Find hidden patterns and associations in customer data
- **AI Optimization**: Use genetic algorithms for hyperparameter tuning
- **Interactive Analytics**: Provide real-time insights through an intuitive dashboard

##  Features

###  Machine Learning Models
- **Logistic Regression**: Linear approach for baseline predictions
- **Decision Trees**: Rule-based interpretable models
- **Random Forest**: Ensemble method for improved accuracy
- **Cross-validation**: Robust model evaluation

###  Customer Analytics
- **K-Means Clustering**: Segment customers into meaningful groups
- **Association Rules Mining**: Discover patterns using Apriori algorithm
- **Feature Engineering**: Automatic creation of relevant features
- **Sentiment Analysis**: NLP-based customer sentiment evaluation

###  AI Optimization
- **Genetic Algorithm**: Automated hyperparameter optimization
- **Multi-objective Optimization**: Balance accuracy, precision, and recall
- **Evolutionary Strategies**: Advanced optimization techniques

###  Interactive Dashboard
- **Real-time Predictions**: Interactive churn probability calculator
- **Data Visualization**: Comprehensive charts and plots
- **Model Performance**: Detailed evaluation metrics
- **Customer Profiling**: Individual customer risk assessment

###  Advanced Analytics
- **Feature Importance**: Understand which factors drive churn
- **ROC Curves**: Visualize model performance
- **Confusion Matrix**: Detailed classification analysis
- **Cross-validation Scores**: Robust performance evaluation

##  Dataset

The project uses [real telecom customer](https://huggingface.co/datasets/aai510-group1/telco-customer-churn) data with the following structure:

```
data/
 train.csv      # Training dataset
 test.csv       # Test dataset for evaluation
 validation.csv # Validation dataset
```

### Key Features
- **Customer Demographics**: Age, gender, location
- **Service Information**: Phone service, internet type, contract details
- **Usage Patterns**: Monthly charges, tenure, service usage
- **Churn Indicators**: Historical churn data and reasons

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/12fahed/Customer-Churn-Prediction-and-Proactive-Retention.git
cd Customer-Churn-Prediction-and-Proactive-Retention
```

### Step 2: Install Dependencies
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn mlxtend textblob deap
```

### Step 3: Prepare Data
Ensure your data files are in the `data/` directory:
- `train.csv` - Training dataset
- `test.csv` - Test dataset (optional)
- `validation.csv` - Validation dataset (optional)

##  Usage

### Running the Application
```bash
streamlit run script.py
```

The application will start a local web server (typically at `http://localhost:8501`).

### Navigation Guide

1. ** Home**: Project overview and data loading
2. ** Data Overview**: Explore dataset structure and statistics
3. ** Churn Prediction**: Train and evaluate ML models
4. ** Customer Segmentation**: Analyze customer clusters
5. ** Association Rules**: Discover behavioral patterns
6. ** AI Optimization**: Genetic algorithm optimization
7. ** Model Performance**: Comprehensive model evaluation
8. ** Interactive Prediction**: Real-time churn prediction

### Quick Start Guide

1. **Load Data**: Click " Load Real Dataset" on the home page
2. **Explore Data**: Navigate to " Data Overview" to understand your dataset
3. **Train Models**: Go to " Churn Prediction" and train ML models
4. **Analyze Segments**: Use " Customer Segmentation" for clustering
5. **Optimize**: Run " AI Optimization" for hyperparameter tuning
6. **Predict**: Use " Interactive Prediction" for real-time predictions

##  Technical Architecture

### Core Components

```
 Data Processing
    Data loading and validation
    Missing value handling
    Feature engineering
    Label encoding

 Machine Learning Pipeline
    Model training (LR, DT, RF)
    Cross-validation
    Hyperparameter tuning
    Performance evaluation

 Analytics Engine
    Customer segmentation (K-Means)
    Association rules mining
    Feature importance analysis
    Sentiment analysis

 AI Optimization
    Genetic algorithm implementation
    Multi-objective optimization
    Evolutionary strategies

 User Interface
     Streamlit dashboard
     Interactive visualizations
     Real-time predictions
     Performance monitoring
```

### Technologies Used

- **Frontend**: Streamlit, HTML/CSS
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **NLP**: TextBlob
- **Optimization**: DEAP (Genetic Algorithms)
- **Association Mining**: MLxtend

##  Model Performance

The system provides comprehensive model evaluation including:

### Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

### Validation
- **Cross-validation**: K-fold validation for robust evaluation
- **Test Set Evaluation**: Independent test data performance
- **Feature Importance**: Understanding key churn drivers

### Optimization
- **Genetic Algorithm**: Automated hyperparameter tuning
- **Multi-objective**: Balance multiple performance metrics
- **Population-based**: Explore multiple solution candidates

##  Configuration

### Model Parameters
You can adjust model parameters in the script:

```python
# Logistic Regression
LogisticRegression(random_state=42, max_iter=1000)

# Random Forest
RandomForestClassifier(random_state=42, n_estimators=100)

# K-Means Clustering
KMeans(n_clusters=3, random_state=42)
```

### Feature Engineering
The system automatically creates engineered features:
- **Service Count**: Total number of subscribed services
- **Tenure Groups**: Customer lifetime categories
- **Spending Patterns**: High/low spender classification
- **Contract Analysis**: Contract type indicators

##  Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/12fahed/Customer-Churn-Prediction-and-Proactive-Retention.git

# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/
```

##  Acknowledgments

- **Scikit-learn** for machine learning algorithms
- **Streamlit** for the web application framework
- **Plotly** for interactive visualizations
- **DEAP** for genetic algorithm implementation

##  Contact

For questions, suggestions, or collaborations:

- **Project Repository**: [GitHub](https://github.com/12fahed/Customer-Churn-Prediction-and-Proactive-Retention)
- **Issues**: [Report Issues](https://github.com/12fahed/Customer-Churn-Prediction-and-Proactive-Retention/issues)

---

**Built with  for data-driven customer retention strategies**
