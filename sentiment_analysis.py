import pandas as pd
from data_collection import fetch_reddit_data, fetch_stock_data
from preprocessing import preprocess_reddit_data, vectorize_text
from model_training import train_test_data_split, train_and_evaluate_model, create_model
from config import MAX_POSTS_PER_SECTOR

def fetch_sector_data(sector_name, companies, etf_ticker, subreddit_str, 
                     start_date, end_date, max_posts, reddit_instance):
    """
    Fetches data for a sector only once, to be reused across multiple models.
    Returns preprocessed data and metadata needed for model training.
    """
    print(f"\n{'='*20} Fetching Data for {sector_name} Sector {'='*20}")
    print(f"Companies: {[c['ticker'] for c in companies]}, ETF: {etf_ticker}, Subreddits: r/{subreddit_str}")
    print(f"Period: {start_date} to {end_date}")

    # Step 1: Download sector ETF data & create labels
    print(f"Downloading {etf_ticker} (Sector ETF) stock data...")
    result = fetch_stock_data(etf_ticker, start_date, end_date)
    if result is None:
        print(f"Error: ETF data download failed for {etf_ticker}. Skipping sector analysis.")
        return None
        
    etf_stock_data, daily_labels_map = result

    # Step 2: Fetch Reddit Data for companies in sector
    query_parts = []
    for company in companies:
        query_parts.append(f"\"{company['name']}\"")  # Quoted name
        query_parts.append(f"{company['ticker']}")
        query_parts.append(f"${company['ticker']}")
    search_query = " OR ".join(query_parts)

    reddit_posts = fetch_reddit_data(reddit_instance, subreddit_str, search_query,
                                    start_date, end_date, max_posts)
                                    
    if reddit_posts.empty:
        print(f"No relevant Reddit posts found for {sector_name} sector companies in r/{subreddit_str}. Skipping analysis.")
        return None

    # Step 3: Clean Reddit Text & Assign Labels
    merged_data = preprocess_reddit_data(reddit_posts, daily_labels_map)

    # ML Steps for the sector
    MIN_SAMPLES = 20
    MIN_CLASSES = 2
    if merged_data.empty or merged_data.shape[0] < MIN_SAMPLES or merged_data['Label'].nunique() < MIN_CLASSES:
        print(f"\nError: Insufficient data for {sector_name} sector to proceed with ML " 
              f"({len(merged_data)} samples, {merged_data['Label'].nunique()} classes). Skipping ML.")
        return None

    # Step 4: TF-IDF Vectorization
    X, y, vectorizer = vectorize_text(merged_data)

    # Step 5: Train/Test Split
    split_result = train_test_data_split(X, y)
    if split_result[0] is None:
        return None
    
    # Return all necessary data for model training
    return {
        'X_train': split_result[0],
        'X_test': split_result[1],
        'y_train': split_result[2],
        'y_test': split_result[3],
        'vectorizer': vectorizer,
        'etf_ticker': etf_ticker
    }

def train_sector_model(sector_name, sector_data, model_type):
    """
    Trains a specific model type on previously fetched sector data.
    """
    if sector_data is None:
        print(f"No data available for {sector_name} sector. Skipping model training.")
        return None
    
    print(f"\n{'='*20} Training {model_type} for {sector_name} Sector {'='*20}")
    
    X_train = sector_data['X_train']
    X_test = sector_data['X_test']
    y_train = sector_data['y_train']
    y_test = sector_data['y_test']
    vectorizer = sector_data['vectorizer']
    etf_ticker = sector_data['etf_ticker']
    
    target_names = [f'{etf_ticker} Down (0)', f'{etf_ticker} Up (1)']
    
    try:
        model = create_model(model_type)
        trained_model = train_and_evaluate_model(
            model_type if model_type != 'ANN' else 'Neural Network',
            model, X_train, X_test, y_train, y_test, 
            target_names, vectorizer, sector_name, etf_ticker
        )
        
        print(f"\n{'='*20} Finished {model_type} Analysis for {sector_name} Sector {'='*20}")
        return trained_model
    except Exception as e:
        print(f"Error during model creation/training for {sector_name} with {model_type}: {e}")
        return None

def run_sector_sentiment_analysis(sector_name, companies, etf_ticker, subreddit_str, 
                                 start_date, end_date, max_posts, reddit_instance, model_type='SVM'):
    """
    Legacy function to maintain compatibility. This runs the full pipeline for a single model.
    For better efficiency, use fetch_sector_data followed by train_sector_model for each model type.
    """
    
    print(f"\n{'='*20} Starting {model_type} Analysis for {sector_name} Sector {'='*20}")
    
    # Fetch data
    sector_data = fetch_sector_data(sector_name, companies, etf_ticker, subreddit_str, 
                                  start_date, end_date, max_posts, reddit_instance)
    
    # Train model
    if sector_data:
        return train_sector_model(sector_name, sector_data, model_type)
    return None