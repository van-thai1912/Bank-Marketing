# 🏦 Bank Marketing Campaign Prediction Web Demo

A comprehensive machine learning web application built with Streamlit for predicting bank marketing campaign success. This demo showcases the complete ML pipeline from exploratory data analysis to model predictions.

## 📋 Project Overview

This project uses machine learning to predict whether a customer will subscribe to a bank product based on their demographic and campaign information.

### Key Features:
- **Exploratory Data Analysis**: Comprehensive data visualization and statistical analysis
- **Two ML Scenarios**:
  - Scenario 1: With `duration` feature (Benchmark - ceiling performance)
  - Scenario 2: Without `duration` feature (Realistic - production-ready)
- **Model Training**: Train Logistic Regression models with SMOTE for imbalanced data
- **Interactive Predictions**: Input customer data and get real-time predictions
- **Model Comparison**: Compare performance across different scenarios
- **Feature Importance Analysis**: Understand which features drive predictions
- **Comprehensive Metrics**: ROC-AUC, PR-AUC, F1-Score, Confusion Matrix, and more

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Navigate to the Project_ML directory (current working directory):
```bash
cd Project_ML
```

2. Install required packages in Python venv:
```bash
python -m venv venv
source venv/bin/activate
pip install -r Web\ Demo/requirements.txt
```

3. Ensure the data file exists:
   - The app expects `Models/processed_bank_data.csv` relative to the app location
   - Update the file path in `app.py` if your data is stored elsewhere

### Running the App

```bash
streamlit run Web\ Demo/app.py
```
The app will open in your default browser at `http://localhost:8501`


### Quick Setup
```bash
cd Project_ML/Web\ Demo
bash ./quickstart.bat  # for Windows
# or
bash ./quickstart.sh  # for Linux/Mac
```

## 📱 App Pages

### 1. 🏠 Home
- Project overview and goals
- Dataset statistics
- Key insights about the data
- Introduction to techniques used

### 2. 📊 Data Exploration
- Dataset overview and statistics
- Missing values analysis
- Target variable distribution
- Numerical features distribution
- Correlation analysis
- Categorical feature exploration

### 3. 🤖 Model Training
- Train models for both scenarios
- View comprehensive performance metrics
- Visualizations:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
  - Feature Importance plots
- Classification Report

### 4. 🔮 Make Predictions
- Interactive input form for customer data
- Real-time predictions for new customers
- Prediction probability visualization
- Model explanation based on input values

### 5. 📊 Model Comparison
- Side-by-side comparison of Scenario 1 vs Scenario 2
- Performance metrics comparison
- Visual charts comparing model performance
- Key insights about each scenario

## 🔧 Technical Stack

### Libraries Used:
- **Streamlit**: Interactive web framework
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and metrics
- **Imbalanced-learn**: SMOTE for handling class imbalance
- **Matplotlib & Seaborn**: Data visualization
- **XGBoost**: Gradient boosting (for future enhancements)

### ML Techniques:
- **Preprocessing**: StandardScaler for feature normalization
- **Imbalanced Data**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Time Split**: Train/test split without shuffling to preserve temporal order
- **Models**: Logistic Regression with configurable hyperparameters

## 📊 Dataset Information

### Features:
- **Age-related**: age, age_group
- **Financial**: balance, deposits, loans
- **Campaign**: duration, campaign, pdays, previous
- **Contact Info**: day, month, contact type, education, job, marital status
- **Target**: y (binary: yes/no for subscription)

### Data Characteristics:
- ~41,000 records
- ~20 features (after feature engineering)
- Highly imbalanced target (only ~11% positive class)
- Time series nature with potential temporal drift

## 🔑 Key Insights

### About Duration Feature:
- Scenario 1 includes `duration` (time spent on call)
- Scenario 2 excludes `duration` (realistic scenario)
- Duration shows **data leakage**: customers who talk longer are more likely to subscribe
- ROC-AUC drops significantly (~15-20 percentage points) without duration

### About Model Performance:
- **With Duration** (Benchmark): ROC-AUC ~0.71
- **Without Duration** (Realistic): ROC-AUC ~0.64
- Time drift present: model performance degrades on future data
- SMOTE helps balance classes: improved recall for positive class

### Recommendations:
1. Use Scenario 2 for production deployment
2. Consider ensemble methods (Random Forest, XGBoost) for better performance
3. Retrain models periodically to account for temporal drift
4. Balance precision-recall trade-off based on business needs

## 🎯 Use Cases

1. **Business Intelligence**: Understand which customers are likely to subscribe
2. **Campaign Targeting**: Identify high-probability customers for focused marketing
3. **Model Benchmarking**: Compare ML approaches and validate model assumptions
4. **Educational**: Learn ML pipeline from data to predictions

## 📈 Expected Performance

| Scenario | ROC-AUC | PR-AUC | F1-Score | Use Case |
|----------|---------|--------|----------|----------|
| With Duration | ~0.71 | ~0.40 | ~0.39 | Benchmark/Upper bound |
| Without Duration | ~0.64 | ~0.35 | ~0.35 | Production/Reality |

*Note: Actual values depend on data and random state*

## 🔄 Data Pipeline

```
Raw Data (processed_bank_data.csv)
    ↓
Feature Engineering (is_contacted_before)
    ↓
Train/Test Split (80/20, time-based)
    ↓
Standardization (StandardScaler)
    ↓
SMOTE (on train set only)
    ↓
Model Training (Logistic Regression)
    ↓
Evaluation & Visualization
    ↓
Interactive Predictions
```

## 🛠️ Customization

### Changing the Data Path:
Edit line in `app.py`:
```python
df = pd.read_csv("YOUR_DATA_PATH/processed_bank_data.csv")
```

### Adjusting Model Hyperparameters:
Modify the LogisticRegression initialization:
```python
model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
```

### Adding New Features:
Edit the preprocessing section to add feature engineering steps

## ⚠️ Important Notes

1. **Data Leakage**: The `duration` feature should NOT be used for real predictions as it's only known after the call
2. **Time Drift**: Financial data shows temporal patterns - the model performs worse on future data
3. **Class Imbalance**: The dataset is highly imbalanced; SMOTE helps but doesn't solve the fundamental challenge
4. **Reproducibility**: Set random_state for reproducible results across runs

## 📚 Further Reading

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [SMOTE Paper](https://arxiv.org/pdf/1106.1813.pdf)
- [Class Imbalance Techniques](https://imbalanced-learn.org/)

## 🤝 Contributing

Feel free to extend this demo by:
- Adding more models (Random Forest, XGBoost, etc.)
- Implementing hyperparameter tuning
- Adding SHAP values for model interpretability
- Creating ensemble models
- Adding model persistence (save/load)

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Bank Marketing ML Demo - Built with Streamlit

---

**Last Updated**: May 2026  
**Data Source**: UCI Bank Marketing Dataset (Processed)
