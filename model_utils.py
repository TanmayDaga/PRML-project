"""
Utility functions for working with saved sentiment analysis models.
This module provides functions to list, load, and use previously trained models.
"""

import os
import pandas as pd
import json
from config import MODEL_DIR
from model_training import load_model

def list_saved_models():
    """List all saved models with their metadata"""
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory {MODEL_DIR} does not exist.")
        return []
        
    saved_models = []
    
    # Find all metadata files
    metadata_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_metadata.json')]
    
    for metadata_file in metadata_files:
        try:
            with open(os.path.join(MODEL_DIR, metadata_file), 'r') as f:
                metadata = json.load(f)
                
            # Extract key information
            model_info = {
                'sector_name': metadata.get('sector_name', 'Unknown'),
                'model_name': metadata.get('model_name', 'Unknown'),
                'accuracy': metadata.get('accuracy', 'N/A'),
                'timestamp': metadata.get('timestamp', 'Unknown'),
                'metadata_file': metadata_file
            }
            saved_models.append(model_info)
        except Exception as e:
            print(f"Error reading metadata file {metadata_file}: {e}")
    
    # Sort by sector, then by model name
    saved_models.sort(key=lambda x: (x['sector_name'], x['model_name']))
    
    return saved_models

def display_models_summary():
    """Display a summary of all saved models"""
    saved_models = list_saved_models()
    
    if not saved_models:
        print("No saved models found.")
        return
    
    print(f"\n{'-'*80}")
    print(f"{'SECTOR':<15} {'MODEL TYPE':<20} {'ACCURACY':<10} {'SAVED DATE':<25}")
    print(f"{'-'*80}")
    
    for model in saved_models:
        print(f"{model['sector_name']:<15} {model['model_name']:<20} {model['accuracy']:<10.4f} {model['timestamp']:<25}")
    
    print(f"{'-'*80}")
    print(f"Total models: {len(saved_models)}")

def predict_sentiment(text, sector_name, model_name):
    """
    Use a saved model to predict sentiment for new text data
    
    Args:
        text (str): Text to analyze
        sector_name (str): Name of the sector model to use
        model_name (str): Type of model to use (e.g., 'SVM', 'Naive Bayes')
    
    Returns:
        dict: Prediction results including class and probability if available
    """
    # Load the model and vectorizer
    model, vectorizer, metadata = load_model(sector_name, model_name)
    
    if not model or not vectorizer:
        print(f"Failed to load model for {sector_name} sector, {model_name} type.")
        return None
    
    # Preprocess and vectorize the text
    from preprocessing import clean_text
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        return {'error': 'Text was empty after cleaning'}
    
    # Create feature vector
    X = vectorizer.transform([cleaned_text])
    
    # Make prediction
    try:
        prediction = int(model.predict(X)[0])
        
        result = {
            'text': text,
            'cleaned_text': cleaned_text,
            'prediction': prediction,
            'label': "Positive (Stock Up)" if prediction == 1 else "Negative (Stock Down)"
        }
        
        # Add probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            result['probability'] = float(probabilities[prediction])
        
        return result
    except Exception as e:
        return {'error': f"Error making prediction: {str(e)}"}

def batch_predict(texts, sector_name, model_name):
    """
    Process a batch of texts and return predictions
    
    Args:
        texts (list): List of text strings to analyze
        sector_name (str): Name of the sector model to use
        model_name (str): Type of model to use
        
    Returns:
        list: Prediction results for each text
    """
    # Load the model and vectorizer
    model, vectorizer, metadata = load_model(sector_name, model_name)
    
    if not model or not vectorizer:
        print(f"Failed to load model for {sector_name} sector, {model_name} type.")
        return None
    
    # Preprocess texts
    from preprocessing import clean_text
    cleaned_texts = [clean_text(text) for text in texts]
    cleaned_texts = [text for text in cleaned_texts if text]  # Remove empty texts
    
    if not cleaned_texts:
        return {'error': 'All texts were empty after cleaning'}
    
    # Vectorize
    X = vectorizer.transform(cleaned_texts)
    
    # Make predictions
    try:
        predictions = model.predict(X)
        results = []
        
        # Add probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            
            for i, (text, cleaned, pred) in enumerate(zip(texts, cleaned_texts, predictions)):
                pred_int = int(pred)
                results.append({
                    'text': text,
                    'cleaned_text': cleaned,
                    'prediction': pred_int,
                    'label': "Positive (Stock Up)" if pred_int == 1 else "Negative (Stock Down)",
                    'probability': float(probabilities[i][pred_int])
                })
        else:
            for text, cleaned, pred in zip(texts, cleaned_texts, predictions):
                pred_int = int(pred)
                results.append({
                    'text': text,
                    'cleaned_text': cleaned,
                    'prediction': pred_int,
                    'label': "Positive (Stock Up)" if pred_int == 1 else "Negative (Stock Down)"
                })
                
        return results
    except Exception as e:
        return {'error': f"Error making batch predictions: {str(e)}"}