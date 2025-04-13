"""
Enhanced visualization module for sentiment analysis project
Provides functions for creating various plots to better understand the data and models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA, TruncatedSVD
from config import PLOT_DIR
import json
import glob

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def plot_confusion_matrix(y_true, y_pred, sector_name, model_name):
    """
    Create a stylized confusion matrix heatmap with additional metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        sector_name: Name of the sector being analyzed
        model_name: Name of the model used
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create figure with two parts: confusion matrix and metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1]})
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
    ax1.set_xlabel('Predicted Labels')
    ax1.set_ylabel('True Labels')
    ax1.set_title(f'Confusion Matrix - {sector_name} ({model_name})')
    ax1.set_xticklabels(['Down', 'Up'])
    ax1.set_yticklabels(['Down', 'Up'])
    
    # Plot metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]
    
    ax2.bar(metrics, values, color=sns.color_palette("Set2"))
    ax2.set_ylim([0, 1])
    for i, v in enumerate(values):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center')
    ax2.set_title(f'Model Performance Metrics - {sector_name} ({model_name})')
    
    plt.tight_layout()
    plot_filename = os.path.join(PLOT_DIR, f"{sector_name}_{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(plot_filename)
    plt.close()
    
    return plot_filename

def plot_roc_curve(y_true, y_score, sector_name, model_name):
    """
    Plot Receiver Operating Characteristic (ROC) curve.
    
    Args:
        y_true: Ground truth labels
        y_score: Probability estimates of the positive class
        sector_name: Name of the sector being analyzed
        model_name: Name of the model used
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {sector_name} ({model_name})')
    plt.legend(loc="lower right")
    
    plot_filename = os.path.join(PLOT_DIR, f"{sector_name}_{model_name.replace(' ', '_')}_roc_curve.png")
    plt.savefig(plot_filename)
    plt.close()
    
    return plot_filename

def plot_precision_recall_curve(y_true, y_score, sector_name, model_name):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: Ground truth labels
        y_score: Probability estimates of the positive class
        sector_name: Name of the sector being analyzed
        model_name: Name of the model used
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = np.mean(precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, label=f'Avg precision = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve - {sector_name} ({model_name})')
    plt.legend(loc="lower left")
    
    plot_filename = os.path.join(PLOT_DIR, f"{sector_name}_{model_name.replace(' ', '_')}_precision_recall.png")
    plt.savefig(plot_filename)
    plt.close()
    
    return plot_filename

def visualize_feature_space(X, y, vectorizer, sector_name, model_name, method='PCA'):
    """
    Visualize text data in 2D space using dimensionality reduction.
    
    Args:
        X: Feature matrix
        y: Labels
        vectorizer: TF-IDF vectorizer used
        sector_name: Name of the sector being analyzed
        model_name: Name of the model used
        method: Dimensionality reduction method ('PCA' or 'SVD')
    """
    # Apply dimensionality reduction
    if method == 'PCA':
        reducer = PCA(n_components=2)
        try:
            X_dense = X.toarray()
            X_reduced = reducer.fit_transform(X_dense)
            explained_var = reducer.explained_variance_ratio_
            title_method = 'PCA'
        except (TypeError, MemoryError):
            # Fallback to SVD if memory error with dense array
            method = 'SVD'
    
    if method == 'SVD':
        reducer = TruncatedSVD(n_components=2)
        X_reduced = reducer.fit_transform(X)
        explained_var = reducer.explained_variance_ratio_
        title_method = 'Truncated SVD'
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='coolwarm', alpha=0.5)
    plt.colorbar(scatter, label='Sentiment (0=Down, 1=Up)')
    
    plt.title(f'{title_method} Visualization of Text Data - {sector_name} ({model_name})')
    plt.xlabel(f'Component 1 ({explained_var[0]:.2%} variance)')
    plt.ylabel(f'Component 2 ({explained_var[1]:.2%} variance)')
    
    # Add important feature annotations
    if hasattr(reducer, 'components_') and hasattr(vectorizer, 'get_feature_names_out'):
        feature_names = vectorizer.get_feature_names_out()
        # Get top features for each principal component
        for i, component in enumerate(reducer.components_):
            top_features_idx = component.argsort()[-10:]  # Get indices of top 10 features
            top_features = [(feature_names[idx], component[idx]) for idx in top_features_idx]
            
            print(f"Top features for component {i+1}:")
            for feature, weight in top_features:
                print(f"  {feature}: {weight:.4f}")
    
    plt.tight_layout()
    
    plot_filename = os.path.join(PLOT_DIR, f"{sector_name}_{model_name.replace(' ', '_')}_feature_space.png")
    plt.savefig(plot_filename)
    plt.close()
    
    return plot_filename

def plot_sentiment_distribution(data, sector_name):
    """
    Plot sentiment distribution and its relationship with ETF movements.
    
    Args:
        data: DataFrame containing dates and labels
        sector_name: Name of the sector being analyzed
    """
    if 'Date' not in data.columns or 'Label' not in data.columns:
        print("Error: Data must contain 'Date' and 'Label' columns")
        return None
    
    # Ensure Date is datetime type
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Aggregate sentiment by day
    daily_sentiment = data.groupby('Date')['Label'].agg(['count', 'mean']).reset_index()
    daily_sentiment.columns = ['Date', 'Post_Count', 'Positive_Ratio']
    
    # Plot time series of post count and sentiment
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Post count over time
    ax1.bar(daily_sentiment['Date'], daily_sentiment['Post_Count'], 
           color=sns.color_palette("Blues_d")[3], alpha=0.7)
    ax1.set_ylabel('Number of Posts')
    ax1.set_title(f'Reddit Post Volume - {sector_name}')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    # Sentiment ratio over time
    ax2.plot(daily_sentiment['Date'], daily_sentiment['Positive_Ratio'], 
            color='green', marker='o', linestyle='-', linewidth=2, markersize=5)
    ax2.axhline(y=0.5, color='red', linestyle='--')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Positive Sentiment Ratio')
    ax2.set_xlabel('Date')
    ax2.set_title(f'Reddit Sentiment Trends - {sector_name}')
    
    plt.tight_layout()
    plot_filename = os.path.join(PLOT_DIR, f"{sector_name}_sentiment_trends.png")
    plt.savefig(plot_filename)
    plt.close()
    
    return plot_filename

def plot_feature_correlations(X, y, vectorizer, sector_name, top_n=15):
    """
    Plot feature correlations with the target variable.
    
    Args:
        X: Feature matrix
        y: Labels
        vectorizer: TF-IDF vectorizer
        sector_name: Name of the sector being analyzed
        top_n: Number of top features to show
    """
    if not hasattr(vectorizer, 'get_feature_names_out'):
        print("Error: Vectorizer must have get_feature_names_out method")
        return None
        
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate correlation coefficient for each feature
    feature_correlations = []
    X_array = X.toarray()
    
    for i in range(X_array.shape[1]):
        correlation = np.corrcoef(X_array[:, i], y)[0, 1]
        if not np.isnan(correlation):
            feature_correlations.append((feature_names[i], correlation))
    
    # Sort by absolute correlation
    feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Plot top correlations
    top_features = feature_correlations[:top_n]
    features, correlations = zip(*top_features)
    
    plt.figure(figsize=(12, 8))
    colors = ['green' if c > 0 else 'red' for c in correlations]
    plt.barh(range(len(features)), correlations, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Correlation with Target (Positive/Negative)')
    plt.title(f'Top {top_n} Features by Correlation with Sentiment - {sector_name}')
    plt.axvline(x=0, color='black', linestyle='--')
    
    plt.tight_layout()
    plot_filename = os.path.join(PLOT_DIR, f"{sector_name}_feature_correlations.png")
    plt.savefig(plot_filename)
    plt.close()
    
    return plot_filename

def compare_models_performance(results_dict, sector_name):
    """
    Create bar chart comparing performance across different models.
    
    Args:
        results_dict: Dictionary with model names as keys and performance metrics as values
        sector_name: Name of the sector being analyzed
    """
    model_names = list(results_dict.keys())
    accuracies = [results_dict[model]['accuracy'] for model in model_names]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(model_names, accuracies, color=sns.color_palette("Set2"))
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.ylim(0, max(accuracies) + 0.1)  # Add some headroom for labels
    plt.xlabel('Model Type')
    plt.ylabel('Accuracy')
    plt.title(f'Model Performance Comparison - {sector_name}')
    
    plt.tight_layout()
    plot_filename = os.path.join(PLOT_DIR, f"{sector_name}_model_comparison.png")
    plt.savefig(plot_filename)
    plt.close()
    
    return plot_filename

def generate_model_comparison_plots():
    """
    Generate comparison plots for all sectors by reading model metadata from saved_models directory.
    This function should be called after all models have been trained.
    """
    print("Generating model comparison plots...")
    
    # Get all sectors from the saved model files
    model_files = glob.glob(os.path.join('saved_models', '*_metadata.json'))
    sectors = set()
    for file in model_files:
        # Extract sector name from filename (format: Sector_ModelType_metadata.json)
        filename = os.path.basename(file)
        sector = filename.split('_')[0]
        sectors.add(sector)
    
    # Generate comparison plot for each sector
    for sector in sectors:
        print(f"Generating comparison plot for {sector} sector")
        sector_models = {}
        
        # Find all model metadata files for this sector
        metadata_files = glob.glob(os.path.join('saved_models', f'{sector}_*_metadata.json'))
        
        for file in metadata_files:
            try:
                with open(file, 'r') as f:
                    metadata = json.load(f)
                
                # Extract model type (e.g., "SVM", "Naive_Bayes", etc.)
                model_type = os.path.basename(file).replace(f"{sector}_", "").replace("_metadata.json", "")
                # Make the name more readable
                model_type = model_type.replace("_", " ")
                
                sector_models[model_type] = {
                    'accuracy': metadata.get('accuracy', 0)
                }
            except Exception as e:
                print(f"Error reading metadata file {file}: {e}")
        
        if sector_models:
            try:
                compare_models_performance(sector_models, sector)
                print(f"Successfully generated comparison plot for {sector}")
            except Exception as e:
                print(f"Error generating comparison plot for {sector}: {e}")
        else:
            print(f"No model data found for {sector}")
    
    print("Model comparison plot generation complete.")