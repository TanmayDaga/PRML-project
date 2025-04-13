import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from config import PLOT_DIR, MODEL_DIR
import joblib
import json
from visualization import (
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_precision_recall_curve,
    visualize_feature_space,
    plot_feature_correlations
)

def train_test_data_split(X, y):
    """Split data into training and testing sets"""
    print("Splitting data into train and test sets...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        if X_test.shape[0] == 0: 
            raise ValueError("Test set is empty")
        return X_train, X_test, y_train, y_test
    except ValueError as e:
        print(f"\nError during train/test split: {e}. Cannot proceed with model training.")
        return None, None, None, None

def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test, target_names, vectorizer, sector_name, etf_ticker):
    """Trains and evaluates a given machine learning model."""
    print(f"\nTraining {model_name} model for {sector_name} sector...")
    model.fit(X_train.toarray(), y_train)
    print(f"{model_name} training complete.")

    print(f"\nEvaluating {model_name} model for {sector_name} sector...")
    y_pred = model.predict(X_test.toarray())

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Generate enhanced visualizations
    print(f"\nGenerating visualizations for {sector_name} sector ({model_name})...")
    
    # 1. Confusion matrix with metrics
    plot_confusion_matrix(y_test, y_pred, sector_name, model_name)
    
    # 2. Feature correlations
    plot_feature_correlations(X_test, y_test, vectorizer, sector_name)
    
    # 3. Feature space visualization using dimensionality reduction
    try:
        visualize_feature_space(X_test, y_test, vectorizer, sector_name, model_name, 
                              method='SVD' if X_test.shape[1] > 50 else 'PCA')
    except Exception as e:
        print(f"Warning: Could not generate feature space visualization: {e}")
    
    # 4. ROC curve and Precision-Recall curve (if model supports probability estimation)
    if hasattr(model, 'predict_proba'):
        try:
            y_score = model.predict_proba(X_test.toarray())[:, 1]
            plot_roc_curve(y_test, y_score, sector_name, model_name)
            plot_precision_recall_curve(y_test, y_score, sector_name, model_name)
        except Exception as e:
            print(f"Warning: Could not generate ROC or PR curves: {e}")
    
    # Also keep the original feature importance plots for specific models
    if model_name == "Naive Bayes":
        plot_nb_features(model, vectorizer, sector_name, etf_ticker)
    elif model_name == "Linear Discriminant Analysis":
        plot_lda_features(model, vectorizer, sector_name)
    elif model_name == "SVM":
        plot_svm_features(model, vectorizer, sector_name, etf_ticker)
        
    # Save the trained model and vectorizer
    save_model(model, vectorizer, model_name, sector_name, accuracy)
    
    return model

def save_model(model, vectorizer, model_name, sector_name, accuracy):
    """Save the trained model, vectorizer, and metadata for later use"""
    # Create a unique filename for the model
    safe_model_name = model_name.replace(" ", "_")
    model_filename = f"{sector_name}_{safe_model_name}.joblib"
    vectorizer_filename = f"{sector_name}_{safe_model_name}_vectorizer.joblib"
    metadata_filename = f"{sector_name}_{safe_model_name}_metadata.json"
    
    # Save the model and vectorizer
    model_path = os.path.join(MODEL_DIR, model_filename)
    vectorizer_path = os.path.join(MODEL_DIR, vectorizer_filename)
    metadata_path = os.path.join(MODEL_DIR, metadata_filename)
    
    try:
        # Save model and vectorizer using joblib
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "sector_name": sector_name,
            "accuracy": float(accuracy),
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_path": model_path,
            "vectorizer_path": vectorizer_path
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Model saved successfully to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        print(f"Model metadata saved to {metadata_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def plot_nb_features(model, vectorizer, sector_name, etf_ticker):
    """Plot feature importances for Naive Bayes model"""
    try:
        feature_names = vectorizer.get_feature_names_out()
        if hasattr(model, 'feature_log_prob_'):
            import numpy as np
            pos_class_prob_sorted = np.argsort(model.feature_log_prob_[1])[:-16:-1]
            neg_class_prob_sorted = np.argsort(model.feature_log_prob_[0])[:-16:-1]

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.barh(np.array(feature_names)[pos_class_prob_sorted], np.exp(model.feature_log_prob_[1][pos_class_prob_sorted]), color='green')
            plt.title(f'Top 15 Features for {etf_ticker} Up Days ({sector_name} - NB)')
            plt.xlabel('Probability')
            plt.gca().invert_yaxis()

            plt.subplot(1, 2, 2)
            plt.barh(np.array(feature_names)[neg_class_prob_sorted], np.exp(model.feature_log_prob_[0][neg_class_prob_sorted]), color='red')
            plt.title(f'Top 15 Features for {etf_ticker} Down Days ({sector_name} - NB)')
            plt.xlabel('Probability')
            plt.gca().invert_yaxis()

            plt.tight_layout()
            plot_filename = os.path.join(PLOT_DIR, f"{sector_name}_nb_features.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved Naive Bayes feature plot to {plot_filename}")
    except Exception as e:
        print(f"Error plotting Naive Bayes features for {sector_name}: {e}")

def plot_lda_features(model, vectorizer, sector_name):
    """Plot feature importances for LDA model"""
    try:
        if hasattr(model, 'coef_'):
            feature_names = vectorizer.get_feature_names_out()
            coefs = model.coef_.flatten()
            feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': coefs})
            feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, 10))
            top_n = 15
            top_pos = feature_importance.head(top_n)
            top_neg = feature_importance.tail(top_n).sort_values(by='Importance')

            plt.subplot(2, 1, 1)
            plt.barh(top_pos['Feature'], top_pos['Importance'], color='green')
            plt.title(f'Top {top_n} Positive Features ({sector_name} - LDA)')
            plt.gca().invert_yaxis()

            plt.subplot(2, 1, 2)
            plt.barh(top_neg['Feature'], top_neg['Importance'], color='red')
            plt.title(f'Top {top_n} Negative Features ({sector_name} - LDA)')
            plt.gca().invert_yaxis()

            plt.tight_layout()
            plot_filename = os.path.join(PLOT_DIR, f"{sector_name}_lda_features.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved LDA feature plot to {plot_filename}")
    except Exception as e:
        print(f"Error plotting LDA features for {sector_name}: {e}")

def plot_svm_features(model, vectorizer, sector_name, etf_ticker):
    """Plot feature importances for SVM model"""
    try:
        if hasattr(model, 'coef_') and model.kernel == 'linear':
            feature_names = vectorizer.get_feature_names_out()
            if hasattr(model.coef_, 'toarray'):
                coefs = model.coef_.toarray().flatten()
            else:
                coefs = model.coef_.flatten()
                
            coef_df = pd.DataFrame({'Feature': feature_names, 'Weight': coefs}).sort_values('Weight', ascending=False)

            # Plot top positive
            plt.figure(figsize=(10, 6))
            top_pos = coef_df.head(15)
            if not top_pos.empty:
                plt.barh(top_pos['Feature'], top_pos['Weight'], color='green')
                plt.title(f'Top 15 Features for {etf_ticker} Up Days ({sector_name} Sector - SVM)')
                plt.xlabel('SVM Weight')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plot_filename_pos = os.path.join(PLOT_DIR, f"{sector_name}_svm_positive_features.png")
                plt.savefig(plot_filename_pos)
                plt.close()
                print(f"Saved positive feature plot to {plot_filename_pos}")
            else:
                print(f"No significant positive features found for {sector_name} (SVM)")

            # Plot top negative
            plt.figure(figsize=(10, 6))
            top_neg = coef_df.tail(15).sort_values('Weight', ascending=True)
            if not top_neg.empty:
                plt.barh(top_neg['Feature'], top_neg['Weight'], color='red')
                plt.title(f'Top 15 Features for {etf_ticker} Down Days ({sector_name} Sector - SVM)')
                plt.xlabel('SVM Weight')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plot_filename_neg = os.path.join(PLOT_DIR, f"{sector_name}_svm_negative_features.png")
                plt.savefig(plot_filename_neg)
                plt.close()
                print(f"Saved negative feature plot to {plot_filename_neg}")
            else:
                print(f"No significant negative features found for {sector_name} (SVM)")
        else:
            print("Plotting requires linear kernel SVM and model coefficients.")
    except Exception as e:
        print(f"Error plotting features for {sector_name} (SVM): {e}")

def create_model(model_type):
    """Create a model instance based on the specified type"""
    if model_type == 'LDA':
        return LinearDiscriminantAnalysis()
    elif model_type == 'Naive Bayes':
        return MultinomialNB()
    elif model_type == 'ANN':
        return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=300, random_state=42)
    elif model_type == 'SVM':
        return SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_model(sector_name, model_name):
    """Load a previously saved model and its vectorizer"""
    safe_model_name = model_name.replace(" ", "_")
    model_filename = f"{sector_name}_{safe_model_name}.joblib"
    vectorizer_filename = f"{sector_name}_{safe_model_name}_vectorizer.joblib"
    metadata_filename = f"{sector_name}_{safe_model_name}_metadata.json"
    
    model_path = os.path.join(MODEL_DIR, model_filename)
    vectorizer_path = os.path.join(MODEL_DIR, vectorizer_filename)
    metadata_path = os.path.join(MODEL_DIR, metadata_filename)
    
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Load metadata if available
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Model accuracy: {metadata.get('accuracy', 'N/A')}")
        
        return model, vectorizer, metadata
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, {}