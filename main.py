#!/usr/bin/env python3
"""
Tesla Stock Sentiment Analysis using Reddit Data
Main entry point for the sentiment analysis project
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from data_collection import initialize_reddit, fetch_sector_data
from sentiment_analysis import train_sector_model
from config import SECTOR_CONFIG, MAX_POSTS_PER_SECTOR
from visualization import generate_model_comparison_plots

def validate_environment():
    """Check if required packages are installed"""
    try:
        import yfinance
        import pandas
        import matplotlib
        import praw
        import sklearn
        print("All required packages are installed.")
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages with:")
        print("pip install yfinance pandas matplotlib praw scikit-learn")
        return False

def main():
    """Main function to run the sentiment analysis pipeline"""
    parser = argparse.ArgumentParser(description='Run sentiment analysis on Reddit posts for stock prediction')
    parser.add_argument('--refresh-data', action='store_true', help='Re-fetch data from Reddit instead of using cached data')
    parser.add_argument('--sectors', nargs='+', help='Specific sectors to analyze (default: all sectors)')
    parser.add_argument('--days', type=int, default=180, help='Number of days to fetch data for (default: 180)')
    parser.add_argument('--max-posts', type=int, default=5000, help='Maximum posts per sector (default: 5000)')
    parser.add_argument('--model-types', nargs='+', default=['SVM', 'LDA', 'Naive_Bayes', 'Neural_Network'], 
                        help='Model types to train (default: all)')
    
    args = parser.parse_args()
    
    print("\n===== Tesla Stock Sentiment Analysis using Reddit Data =====")
    print("Checking environment...")
    
    if not validate_environment():
        sys.exit(1)
        
    # Initialize Reddit API connection
    reddit_instance = initialize_reddit()
    if not reddit_instance:
        print("FATAL: Cannot initialize Reddit connection. Exiting.")
        sys.exit(1)
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Set up sectors to analyze
    sectors_to_analyze = args.sectors if args.sectors else list(SECTOR_CONFIG.keys())
    
    print(f"Starting analysis for sectors: {', '.join(sectors_to_analyze)}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Define models to run
    model_types = args.model_types
    all_sector_models = {model_type: {} for model_type in model_types}
    
    # Process each sector once, then apply all model types
    print(f"\n{'#'*30} FETCHING DATA FOR ALL SECTORS {'#'*30}")
    sector_data_cache = {}  # Store fetched data for each sector
    
    # First, fetch data for all sectors
    for sector_name, config in SECTOR_CONFIG.items():
        if sector_name not in sectors_to_analyze:
            print(f"Skipping sector {sector_name} as it is not in the list of sectors to analyze.")
            continue
        
        # Combine multiple subreddits with "+" for Reddit API
        subreddit_list = config.get('subreddits', ['stocks'])
        if not isinstance(subreddit_list, list) or not subreddit_list:
             print(f"Warning: Invalid/empty subreddit list for {sector_name}. Defaulting to 'stocks'.")
             subreddit_list = ['stocks']
             
        subreddit_list = [sub.strip() for sub in subreddit_list if sub.strip()]
        if not subreddit_list:
             print(f"Warning: Subreddit list empty after cleaning for {sector_name}. Defaulting to 'stocks'.")
             subreddit_list = ['stocks']
             
        subreddit_string_combined = "+".join(subreddit_list)
        
        # Fetch data for this sector once
        sector_data = fetch_sector_data(
            sector_name=sector_name,
            companies=config['companies'],
            etf_ticker=config['etf'],
            subreddit_str=subreddit_string_combined,
            start_date=start_date,
            end_date=end_date,
            max_posts=args.max_posts,
            reddit_instance=reddit_instance
        )
        
        if sector_data:
            sector_data_cache[sector_name] = sector_data
            print(f"✓ Successfully fetched data for {sector_name} sector")
        else:
            print(f"✗ Failed to fetch usable data for {sector_name} sector")
    
    # Now run each model type on the cached data
    for model_type in model_types:
        print(f"\n{'#'*30} Starting Analysis with {model_type} {'#'*30}")
        
        for sector_name in sector_data_cache.keys():
            # Train model on the cached data
            model = train_sector_model(
                sector_name=sector_name,
                sector_data=sector_data_cache[sector_name],
                model_type=model_type
            )
            
            if model:
                 all_sector_models[model_type][sector_name] = model
    
    # Print summary of results
    print("\n--- Multi-Sector Analysis Script Finished ---")
    for model_type, sector_models in all_sector_models.items():
        print(f"\nResults for {model_type}:")
        print(f"Attempted analysis for sectors: {list(sector_data_cache.keys())}")
        print(f"Successfully trained models for sectors: {list(sector_models.keys())}")
    
    print("\nCheck console output and saved plot files (.png) in 'sector_plots' directory for results.")
    
    # After all models are trained, generate comparison plots
    generate_model_comparison_plots()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")