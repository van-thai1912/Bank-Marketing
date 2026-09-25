"""
Utility functions for Bank Marketing Demo
Helper functions for data processing, modeling, and visualization
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_validate_data(filepath):
    """
    Load data from CSV and perform basic validation
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data validation fails
    """
    try:
        df = pd.read_csv(filepath)
        
        if df.empty:
            raise ValueError("Data file is empty")
            
        if "y" not in df.columns:
            raise ValueError("Target column 'y' not found in data")
            
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")


def calculate_all_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        dict: Dictionary containing all metrics
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "f1": f1_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def get_feature_importance_top_n(model, feature_names, n=10):
    """
    Get top N most important features based on model coefficients
    
    Args:
        model: Trained model with coef_ attribute
        feature_names: List of feature names
        n: Number of top features to return
        
    Returns:
        DataFrame: Features sorted by coefficient magnitude
    """
    feature_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": model.coef_[0],
        }
    )
    
    # Sort by absolute value
    feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()
    feature_importance = feature_importance.sort_values("Abs_Coefficient", ascending=False)
    
    return feature_importance.head(n)[["Feature", "Coefficient"]]


def plot_confusion_matrix(y_true, y_pred, ax=None, cmap="Blues"):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        ax: Matplotlib axis (if None, creates new)
        cmap: Colormap name
        
    Returns:
        Figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        xticklabels=["No (0)", "Yes (1)"],
        yticklabels=["No (0)", "Yes (1)"],
        ax=ax,
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    
    return fig, ax


def plot_roc_curve(y_true, y_pred_proba, ax=None, label="Model"):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        ax: Matplotlib axis (if None, creates new)
        label: Label for the curve
        
    Returns:
        Figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"{label} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    return fig, ax


def plot_precision_recall_curve(y_true, y_pred_proba, ax=None, label="Model"):
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        ax: Matplotlib axis (if None, creates new)
        label: Label for the curve
        
    Returns:
        Figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    baseline = y_true.mean()
    
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"{label} (AP = {pr_auc:.4f})")
    ax.axhline(y=baseline, color="navy", linestyle="--", lw=1, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    return fig, ax


def format_metrics_for_display(metrics, decimal_places=4):
    """
    Format metrics for display
    
    Args:
        metrics: Dictionary of metrics
        decimal_places: Number of decimal places to round to
        
    Returns:
        dict: Formatted metrics
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and key != "confusion_matrix":
            formatted[key] = round(value, decimal_places)
        else:
            formatted[key] = value
    
    return formatted


def get_prediction_explanation(model, input_data, feature_names, top_n=5):
    """
    Get explanation for a prediction based on feature coefficients
    
    Args:
        model: Trained model with coef_ attribute
        input_data: Input features (scaled)
        feature_names: List of feature names
        top_n: Number of top influential features to return
        
    Returns:
        dict: Explanation including top positive and negative influences
    """
    coefficients = model.coef_[0]
    
    # Calculate influence (coefficient * feature value)
    influences = coefficients * input_data[0]
    
    influence_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Value": input_data[0],
            "Influence": influences,
        }
    )
    
    # Sort by absolute influence
    influence_df["Abs_Influence"] = influence_df["Influence"].abs()
    
    top_positive = influence_df.nlargest(top_n, "Influence")
    top_negative = influence_df.nsmallest(top_n, "Influence")
    
    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
        "total_influence": influences.sum(),
    }


def create_prediction_summary(prediction, probability):
    """
    Create a summary of prediction results
    
    Args:
        prediction: Binary prediction (0 or 1)
        probability: Probability of positive class
        
    Returns:
        dict: Summary information
    """
    return {
        "prediction": "Will Subscribe" if prediction == 1 else "Will Not Subscribe",
        "probability": probability,
        "confidence": max(probability, 1 - probability),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low",
    }
