import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """Cleans text data for analysis."""
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+|www\.\S+|t\.co/\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"/u/\w+|r/\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_reddit_data(reddit_posts, daily_labels_map):
    """Clean Reddit text and assign labels based on stock movement"""
    print(f"\nCleaning {len(reddit_posts)} fetched Reddit posts...")
    reddit_posts['Cleaned_Text'] = reddit_posts['Text'].apply(clean_text)
    
    # Assign labels
    print("Assigning daily stock-based labels to posts...")
    if not daily_labels_map.empty:
        reddit_posts['Timestamp'] = pd.to_datetime(reddit_posts['Timestamp'])
        reddit_posts['Date'] = reddit_posts['Timestamp'].dt.date
        reddit_posts['Label'] = reddit_posts['Date'].map(daily_labels_map)
        unmapped_count = reddit_posts['Label'].isna().sum()
        
        if unmapped_count > 0: 
            print(f"Excluding {unmapped_count} posts from non-trading days.")
        
        reddit_posts.dropna(subset=['Label'], inplace=True)
        
        if not reddit_posts.empty:
            reddit_posts['Label'] = reddit_posts['Label'].astype(int)
            initial_count = len(reddit_posts)
            reddit_posts = reddit_posts[reddit_posts['Cleaned_Text'] != '']
            filtered_count = initial_count - len(reddit_posts)
            
            if filtered_count > 0: 
                print(f"Removed {filtered_count} posts with empty text.")
            
            print(f"Total posts after mapping/cleaning: {len(reddit_posts)}")
            
            if not reddit_posts.empty: 
                print(f"Label distribution:\n{reddit_posts['Label'].value_counts()}")
            else: 
                print("Warning: No posts remained after cleaning.")
        else: 
            print("No posts remained after dropping non-trading days.")
            
    else:
        print("Skipping label assignment: No daily stock labels generated.")
    
    return reddit_posts

def vectorize_text(data, max_features=1000):
    """Convert text to TF-IDF features"""
    print("\nExtracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', min_df=2)
    X = vectorizer.fit_transform(data['Cleaned_Text'])
    y = data['Label']
    print(f"Feature matrix shape: {X.shape}")
    return X, y, vectorizer